const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");

const dtype_o = @import("./dtype.zig");
const DataType = dtype_o.DataType;
const Scalar = dtype_o.Scalar;

const storage_t = @import("./storage.zig");
const Device = storage_t.Device;
const layout_t = @import("./layout.zig");

const F = @import("nn/functional.zig");

pub fn Tensor(comptime N: usize, comptime storage_args: struct {
    T: type = f32,
    comptime D: Device = .Cpu,
}) type {
    return struct {
        const StorageArgs = storage_args;
        const Storage = storage_t.Storage(storage_args.T, storage_args.D);
        const ShapeIterator = layout_t.ShapeIterator(N);

        pub const T = storage_args.T;
        pub const D = storage_args.D;
        pub const DIMS = N;

        const Layout = layout_t.Layout(N);

        const Self = @This();

        _base: ?*const Self = null,
        storage: Storage,
        layout: Layout,
        _storage_offset: usize = 0,

        // scope method
        // divide
        pub fn split(self: *const Self, allocator: std.mem.Allocator, chunk_size: usize, dim: usize) ![]const Self {
            if (dim >= self.ndim()) return error.InvalidDim;

            const dim_len = self.shape()[dim];
            if (chunk_size == 0 or chunk_size > dim_len) return error.InvalidSplit;

            const num_splits = (dim_len + chunk_size - 1) / chunk_size;
            var result = try allocator.alloc(Self, num_splits);

            var offset: usize = 0;
            for (0..num_splits) |i| {
                const chunk_size_i = if ((offset + chunk_size) <= dim_len) chunk_size else (dim_len - offset);

                var new_shape = self.shape();
                new_shape[dim] = chunk_size_i;

                // must use old strides
                const new_strides = self.stride();

                const layout = Layout.initRaw(new_shape, new_strides);
                result[i] = try Self.fromDataImpl(layout, self.storage.shared(), self._storage_offset + offset * self.stride()[dim]);

                offset += chunk_size_i;
            }

            return result;
        }

        pub fn chunk(self: *const Self, allocator: std.mem.Allocator, chunk_count: usize, dim: usize) ![]const Self {
            if (dim >= self.ndim()) return error.InvalidDim;

            const dim_len = self.shape()[dim];
            if (chunk_count == 0 or chunk_count > dim_len) return error.InvalidSplit;

            const chunk_size_i = (dim_len + chunk_count - 1) / chunk_count;
            return try self.split(allocator, chunk_size_i, dim);
        }

        pub fn dataItem(self: *const Self) !T {
            var data_iter = self.shapeIter();
            const item = data_iter.next();

            if (item) |i| {
                return try self.getData(i);
            } else {
                return error.EmptyTensor;
            }
        }

        pub fn unbind(self: *const Self, allocator: std.mem.Allocator, dim: usize) ![]const Tensor(N - 1, storage_args) {
            if (dim >= self.ndim()) return error.InvalidDim;

            const dim_len = self.shape()[dim];

            const NT = Tensor(N - 1, storage_args);

            var result = try allocator.alloc(NT, dim_len);

            var offset: usize = 0;
            for (0..dim_len) |idx| {
                var new_shape = [_]usize{0} ** (N - 1);
                var new_stride = [_]usize{0} ** (N - 1);

                {
                    var i: usize = 0;
                    var j: usize = 0;

                    while (j < N) {
                        if (j == dim) {
                            j += 1;
                        } else {
                            new_shape[i] = self.shape()[j];
                            new_stride[i] = self.stride()[j];

                            i += 1;
                            j += 1;
                        }
                    }
                }

                const layout = layout_t.Layout(N - 1).initRaw(new_shape, new_stride);
                result[idx] = try NT.fromDataImpl(layout, self.storage.shared(), self._storage_offset + offset * self.stride()[dim]);

                offset += 1;
            }

            return result;
        }

        pub fn oneHot(self: *const Self, comptime TT: type, num_classes: ?usize) !Tensor(N + 1, .{ .T = TT }) {
            switch (@typeInfo(T)) {
                .int => {
                    const new_dim = if (num_classes) |nc| nc else blk: {
                        const max_value = try self.maxAll();
                        defer max_value.deinit();

                        const max_v_item = try max_value.dataItem();
                        break :blk @as(usize, @intCast(max_v_item + 1));
                    };

                    var result_tensor = try full(self.s_allocator(), try utils.array.insertDim(N, self.shape(), N, new_dim), @as(TT, 0));

                    var self_iter = self.shapeIter();
                    while (self_iter.next()) |idx| {
                        const val = try self.getData(idx);
                        const dest_idx = try utils.array.insertDim(N, idx, N, @as(usize, @intCast(val)));
                        try result_tensor.setData(dest_idx, @as(TT, 1));
                    }

                    return result_tensor;
                },
                else => @compileError("only support int tensor for oneHot"),
            }
        }

        pub fn pad(self: *const Self, pads: []const usize, value: T) !Self {
            const pad_dims = pads.len / 2;

            var new_shape = self.shape();
            for (&new_shape, 0..) |*s, i| {
                const idx_to_pad_idx = N - i - 1;

                if (idx_to_pad_idx < pad_dims) {
                    const left_add = pads[2 * idx_to_pad_idx];
                    const right_add = pads[2 * idx_to_pad_idx + 1];
                    s.* += left_add + right_add;
                }
            }

            var result = try full(self.s_allocator(), new_shape, value);

            var shape_iter = self.shapeIter();
            while (shape_iter.next()) |idx| {
                var dst_idx = idx;

                for (0..N) |i| {
                    // judge if need to set value from orig tensor
                    const idx_to_pad_idx = N - i - 1;

                    if (idx_to_pad_idx < pad_dims) {
                        const left_add = pads[2 * idx_to_pad_idx];
                        dst_idx[i] += left_add;
                    }
                }

                try result.setData(dst_idx, try self.getData(idx));
            }

            return result;
        }

        pub fn mseLoss(self: Self, other: Self) !Tensor(0, .{ .T = T }) {
            return try F.mseLoss(N, T, self, other);
        }

        pub fn crossEntropy(self: Self, other: Self) !Tensor(0, .{ .T = T }) {
            return try F.crossEntropy(N, T, self, other);
        }

        // elementwise method
        pub fn map_(self: *Self, ctx: anytype, func: fn (T, @TypeOf(ctx)) T) void {
            var iter = self.shapeIter();

            while (iter.next()) |idx| {
                self.setData(idx, func(self.getData(idx) catch unreachable, ctx)) catch unreachable;
            }
        }

        pub fn map(
            self: *const Self,
            comptime RT: type,
            ctx: anytype,
            func: fn (T, @TypeOf(ctx)) RT,
        ) !Tensor(N, .{ .T = RT }) {
            const TI = Tensor(N, .{ .T = RT });

            var new_buf = try self.storage.allocator.alloc(RT, self.size());

            var iter = self.shapeIter();

            var i: usize = 0;
            while (iter.next()) |idx| {
                new_buf[i] = func(self.getData(idx) catch unreachable, ctx);
                i += 1;
            }

            const layout = Layout.init(self.shape());
            const storage = try storage_t.Storage(RT, .Cpu).initImpl(self.storage.allocator, new_buf);

            return try TI.fromDataImpl(layout, storage, 0);
        }

        pub fn eql(self: *const Self, value: T) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, ctx: T) bool {
                    return v == ctx;
                }
            }.call;
            return try self.map(bool, value, func);
        }

        pub fn lt(self: *const Self, value: T) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, ctx: T) bool {
                    return v < ctx;
                }
            }.call;
            return try self.map(bool, value, func);
        }

        pub fn le(self: *const Self, value: T) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, ctx: T) bool {
                    return v <= ctx;
                }
            }.call;
            return try self.map(bool, value, func);
        }

        pub fn gt(self: *const Self, value: T) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, ctx: T) bool {
                    return v > ctx;
                }
            }.call;
            return try self.map(bool, value, func);
        }

        pub fn ge(self: *const Self, value: T) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, ctx: T) bool {
                    return v >= ctx;
                }
            }.call;
            return try self.map(bool, value, func);
        }

        pub fn maskedFill_(self: *Self, mask: Tensor(N, .{ .T = bool }), value: T) !void {
            var mask_i = mask;

            try mask_i.broadcastTo_(self.shape());

            var iter = self.shapeIter();

            while (iter.next()) |idx| {
                if (try mask_i.getData(idx)) {
                    try self.setData(idx, value);
                }
            }
        }

        pub fn binaryOp_(self: *Self, b: Self, op_func: fn (x: T, y: T) T) !void {
            // inplace method: need broadcast to self shape
            var b_i = b;
            try b_i.broadcastTo_(self.shape());

            var iter = self.shapeIter();

            while (iter.next()) |idx| {
                const x = try self.getData(idx);
                const y = try b_i.getData(idx);

                try self.setData(idx, op_func(x, y));
            }
        }

        pub fn binaryOp(
            self: Self,
            b: Self,
            op_func: fn (x: T, y: T) T,
        ) !Self {
            const target_shape = try utils.compatibleBroacastShapes(
                N,
                self.shape(),
                b.shape(),
            );

            std.debug.print("target shape: {any}\n", .{target_shape});

            const a = try self.broadcastTo(target_shape);
            defer a.deinit();
            const c = try b.broadcastTo(target_shape);
            defer c.deinit();

            std.debug.print("a: {f} c: {f}\n", .{ a, c });
            var new_buf = try self.s_allocator().alloc(T, utils.product(&target_shape));

            var iter_a = a.shapeIter();

            var i: usize = 0;

            while (iter_a.next()) |idx| {
                const x = try a.getData(idx);
                const y = try c.getData(idx);

                const flat_idx = try utils.indexShapeToFlat(N, target_shape, idx);

                new_buf[flat_idx] = op_func(x, y);
                i += 1;
            }

            const layout = Layout.init(target_shape);
            const storage = try Storage.initImpl(
                self.s_allocator(),
                new_buf,
            );

            return try Self.fromDataImpl(layout, storage, 0);
        }

        pub fn clamp_(self: *Self, min_a: T, max_a: T) void {
            const ctx_i = .{
                .min = min_a,
                .max = max_a,
            };
            const func = struct {
                fn call(v: T, ctx: @TypeOf(ctx_i)) T {
                    return std.math.clamp(v, ctx.min, ctx.max);
                }
            }.call;

            self.map_(ctx_i, func);
        }

        pub fn add_(self: *Self, value: anytype) !void {
            const TV = @TypeOf(value);

            if (TV == @This()) {
                const func = struct {
                    fn call(v: T, other: T) T {
                        return v + other;
                    }
                }.call;

                return try self.binaryOp_(@as(@This(), value), func);
            }

            if (TV == T or (TV == comptime_float and @typeInfo(T) == .float) or (TV == comptime_int and @typeInfo(T) == .int)) {
                const vv = @as(T, value);

                const func = struct {
                    fn call(v: T, ctx: T) T {
                        return v + ctx;
                    }
                }.call;

                return self.map_(vv, func);
            }

            @compileError("unsupported add_ argument type " ++ @typeName(TV) ++ " for tensor of type " ++ @typeName(T));
        }

        pub fn add(self: *const Self, value: anytype) !Self {
            const TV = @TypeOf(value);
            switch (TV) {
                @This() => {
                    const func = struct {
                        fn call(v: T, other: T) T {
                            return v + other;
                        }
                    }.call;

                    return try self.binaryOp(@as(@This(), value), func);
                },
                T => {
                    const vv = @as(T, value);

                    const func = struct {
                        fn call(v: T, ctx: T) T {
                            return v + ctx;
                        }
                    }.call;

                    return try self.map(T, vv, func);
                },
                else => @compileError("unsupported add argument type" ++ " self: " ++ @typeName(@This()) ++ " input: " ++ @typeName(TV)),
            }
        }

        pub fn sub_(self: *Self, value: anytype) !void {
            const TV = @TypeOf(value);

            if (TV == @This()) {
                const func = struct {
                    fn call(v: T, other: T) T {
                        return v - other;
                    }
                }.call;

                return try self.binaryOp_(@as(@This(), value), func);
            }

            if (TV == T or (TV == comptime_float and @typeInfo(T) == .float) or (TV == comptime_int and @typeInfo(T) == .int)) {
                const vv = @as(T, value);

                const func = struct {
                    fn call(v: T, ctx: T) T {
                        return v - ctx;
                    }
                }.call;

                return self.map_(vv, func);
            }

            @compileError("unsupported sub_ argument type " ++ @typeName(TV) ++ " for tensor of type " ++ @typeName(T));
        }

        pub fn sub(self: *const Self, value: anytype) !Self {
            const TV = @TypeOf(value);
            switch (TV) {
                @This() => {
                    const func = struct {
                        fn call(v: T, other: T) T {
                            return v - other;
                        }
                    }.call;

                    return try self.binaryOp(@as(@This(), value), func);
                },
                T => {
                    const vv = @as(T, value);

                    const func = struct {
                        fn call(v: T, ctx: T) T {
                            return v - ctx;
                        }
                    }.call;

                    return try self.map(T, vv, func);
                },
                else => @compileError("unsupported sub argument type" ++ " self: " ++ @typeName(@This()) ++ " input: " ++ @typeName(TV)),
            }
        }

        pub fn mul_(self: *Self, value: anytype) !void {
            const TV = @TypeOf(value);

            if (TV == @This()) {
                const func = struct {
                    fn call(v: T, other: T) T {
                        return v * other;
                    }
                }.call;

                return try self.binaryOp_(@as(@This(), value), func);
            }

            if (TV == T or (TV == comptime_float and @typeInfo(T) == .float) or (TV == comptime_int and @typeInfo(T) == .int)) {
                const vv = @as(T, value);

                const func = struct {
                    fn call(v: T, ctx: T) T {
                        return v * ctx;
                    }
                }.call;

                return self.map_(vv, func);
            }

            @compileError("unsupported mul_ argument type " ++ @typeName(TV) ++ " for tensor of type " ++ @typeName(T));
        }

        pub fn mul(self: *const Self, value: anytype) !Self {
            const TV = @TypeOf(value);
            switch (TV) {
                @This() => {
                    const func = struct {
                        fn call(v: T, other: T) T {
                            return v * other;
                        }
                    }.call;

                    return try self.binaryOp(@as(@This(), value), func);
                },
                T => {
                    const vv = @as(T, value);

                    const func = struct {
                        fn call(v: T, ctx: T) T {
                            return v * ctx;
                        }
                    }.call;

                    return try self.map(T, vv, func);
                },
                else => @compileError("unsupported mul argument type" ++ " self: " ++ @typeName(@This()) ++ " input: " ++ @typeName(TV)),
            }
        }

        pub fn div_(self: *Self, value: anytype) !void {
            const TV = @TypeOf(value);

            if (TV == @This()) {
                const func = struct {
                    fn call(v: T, other: T) T {
                        return v / other;
                    }
                }.call;

                return try self.binaryOp_(@as(@This(), value), func);
            }

            if (TV == T or ((TV == comptime_float or TV == comptime_int) and @typeInfo(T) == .float) or (TV == comptime_int and @typeInfo(T) == .int)) {
                const vv = @as(T, value);

                const func = struct {
                    fn call(v: T, ctx: T) T {
                        return v / ctx;
                    }
                }.call;

                return self.map_(vv, func);
            }

            @compileError("unsupported div_ argument type " ++ @typeName(TV) ++ " for tensor of type " ++ @typeName(T));
        }

        pub fn div(self: *const Self, value: anytype) !Self {
            const TV = @TypeOf(value);
            switch (TV) {
                @This() => {
                    const func = struct {
                        fn call(v: T, other: T) T {
                            return v / other;
                        }
                    }.call;

                    return try self.binaryOp(@as(@This(), value), func);
                },
                T => {
                    const vv = @as(T, value);

                    const func = struct {
                        fn call(v: T, ctx: T) T {
                            return v / ctx;
                        }
                    }.call;

                    return try self.map(T, vv, func);
                },
                else => @compileError("unsupported div argument type" ++ " self: " ++ @typeName(@This()) ++ " input: " ++ @typeName(TV)),
            }
        }

        pub fn sin_(self: *Self) void {
            if (@typeInfo(T) != .float) @compileError("only supported float tensor sin_ op");

            const func = struct {
                fn call(v: T, _: void) T {
                    return @sin(v);
                }
            }.call;
            self.map_(void{}, func);
        }

        pub fn exp_(self: *Self) void {
            const func = struct {
                fn call(v: T, _: void) T {
                    return @exp(v);
                }
            }.call;
            return self.map_(void{}, func);
        }

        pub fn log_(self: *Self) void {
            const func = struct {
                fn call(v: T, _: void) T {
                    return @log(v);
                }
            }.call;
            return self.map_(void{}, func);
        }

        pub fn sigmoid_(self: *Self) void {
            if (@typeInfo(T) != .float) @compileError("only supported float tensor sin_ op");

            const func = struct {
                fn call(v: T, _: void) T {
                    return 1.0 / (1.0 + @exp(-v));
                }
            }.call;
            return self.map_(void{}, func);
        }

        pub fn relu_(self: *Self) void {
            const func = struct {
                fn call(v: T, _: void) T {
                    return @max(v, @as(T, 0));
                }
            }.call;
            return self.map_(void{}, func);
        }

        pub fn powi_(self: *Self, value: T) void {
            const func = struct {
                fn call(v: T, ctx: T) T {
                    return std.math.pow(T, v, ctx);
                }
            }.call;
            return self.map_(value, func);
        }

        pub fn sqrt_(self: *Self) void {
            if (@typeInfo(T) != .float) @compileError("only supported float tensor sin_ op");

            const func = struct {
                fn call(v: T, _: void) T {
                    return @sqrt(v);
                }
            }.call;
            return self.map_(void{}, func);
        }

        // check
        pub fn isNan(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isNan(v);
                }
            }.call;
            return try self.map(bool, void{}, func);
        }

        pub fn isInf(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isInf(v);
                }
            }.call;
            return try self.map(bool, void{}, func);
        }

        pub fn isPositiveInf(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isPositiveInf(v);
                }
            }.call;
            return try self.map(bool, void{}, func);
        }

        pub fn isNegativeInf(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isNegativeInf(v);
                }
            }.call;
            return try self.map(bool, void{}, func);
        }

        pub fn isFinite(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isFinite(v);
                }
            }.call;
            return try self.map(bool, void{}, func);
        }

        pub fn nanToNum_(self: *Self, args: struct {
            nan: T,
            posinf: ?T = null,
            neginf: ?T = null,
        }) void {
            const Ctx = struct {
                nan: T,
                posinf: ?T,
                neginf: ?T,
            };

            const ctx_i = Ctx{
                .nan = args.nan,
                .posinf = if (args.posinf) |posinf| posinf else null,
                .neginf = if (args.neginf) |neginf| neginf else null,
            };

            const func = struct {
                fn call(v: T, ctx: Ctx) T {
                    if (std.math.isNan(v)) {
                        return ctx.nan;
                    } else if (std.math.isPositiveInf(v)) {
                        return if (ctx.posinf) |posinf| posinf else v;
                    } else if (std.math.isNegativeInf(v)) {
                        return if (ctx.neginf) |neginf| neginf else v;
                    } else {
                        return v;
                    }
                }
            }.call;
            return self.map_(ctx_i, func);
        }

        pub fn softmax(self: *const Self) !Self {
            const dims = self.ndim();

            if (dims == 0) {
                return error.InvalidDimension;
            }

            const a = try self.max(N - 1);
            defer a.deinit();

            std.debug.print("self: {f} a: {f}\n", .{ self, a });
            var v = try self.sub(a);
            v.exp_();

            const vs = try v.sum(N - 1);
            defer vs.deinit();

            std.debug.print("v: {f} vs: {f}\n", .{ v, vs });
            try v.div_(vs);

            return v;
        }

        // reduce method
        pub fn reduce(
            self: *const Self,
            dm: usize,
            op_func: fn (acc: T, x: T) T,
            comptime PT: type,
            post_func: ?fn (acc: T, count: usize) PT,
        ) !Tensor(N, .{ .T = PT }) {
            var shape_i = self.shape();
            shape_i[dm] = 1;

            const data_len = utils.product(&shape_i);
            var new_buf = try self.s_allocator().alloc(PT, data_len);

            var shape_i_iter = layout_t.initShapeIterator(shape_i);
            while (shape_i_iter.next()) |idx| {
                var acc = blk: {
                    if (self.shape()[dm] == 1) {
                        break :blk try self.getData(idx);
                    } else {
                        var idx_i = idx;
                        var idx_i_1 = idx;
                        idx_i_1[dm] = 1;

                        var acc = op_func(try self.getData(idx_i), try self.getData(idx_i_1));

                        for (2..self.shape()[dm]) |k| {
                            idx_i[dm] = k;
                            acc = op_func(acc, try self.getData(idx_i));
                        }
                        break :blk acc;
                    }
                };

                if (post_func) |pf| {
                    acc = pf(acc, self.shape()[dm]);
                }

                const flat_idx = try utils.indexShapeToFlat(N, shape_i, idx);
                new_buf[flat_idx] = acc;
            }

            const layout = Layout.init(shape_i);
            const storage = try storage_t.Storage(PT, .Cpu).initImpl(self.s_allocator(), new_buf);

            return try Tensor(N, .{ .T = PT }).fromDataImpl(layout, storage, 0);
        }

        pub fn reduceAll(
            self: *const Self,
            op_func: fn (acc: T, x: T) T,
            comptime PT: type,
            post_func: ?fn (acc: T, count: usize) PT,
        ) !Tensor(0, .{ .T = PT }) {
            var shape_iter = ShapeIterator.init(self.shape());

            const idx0 = shape_iter.next().?;
            const idx1 = shape_iter.next().?;

            var acc = op_func(try self.getData(idx0), try self.getData(idx1));

            while (shape_iter.next()) |idx| {
                acc = op_func(acc, try self.getData(idx));
            }

            const new_buf = try self.s_allocator().alloc(PT, 1);

            const layout = layout_t.Layout(0).init([_]usize{});

            if (post_func) |pf| {
                const count = self.size();
                const posted_v = pf(acc, count);

                new_buf[0] = posted_v;
            } else {
                new_buf[0] = acc;
            }

            const storage = try storage_t.Storage(PT, .Cpu).initImpl(self.s_allocator(), new_buf);

            return try Tensor(0, .{ .T = PT }).fromDataImpl(layout, storage, 0);
        }

        pub fn sum(self: *const Self, dim: usize) !Self {
            const func = struct {
                fn op_func(acc: T, val: T) T {
                    return acc + val;
                }
            }.op_func;
            return try self.reduce(dim, func, T, null);
        }

        pub fn sumAll(self: *const Self) !Tensor(0, .{ .T = T }) {
            const func = struct {
                fn op_func(acc: T, val: T) T {
                    return acc + val;
                }
            }.op_func;
            return try self.reduceAll(func, T, null);
        }

        pub fn max(self: *const Self, dim: usize) !Self {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return @max(acc, val);
                }
            };
            return try self.reduce(dim, scope.op_func, T, null);
        }

        pub fn maxAll(self: *const Self) !Tensor(0, .{ .T = T }) {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return @max(acc, val);
                }
            };
            return try self.reduceAll(scope.op_func, T, null);
        }

        pub fn min(self: *const Self, dim: usize) !Self {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return @min(acc, val);
                }
            };
            return try self.reduce(dim, scope.op_func, T, null);
        }

        pub fn minAll(self: *const Self) !Self {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return @min(acc, val);
                }
            };
            return try self.reduceAll(scope.op_func, T, null);
        }

        pub fn prod(self: *const Self, dim: usize) !Self {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return acc * val;
                }
            };
            return try self.reduce(dim, scope.op_func, T, null);
        }

        pub fn prodAll(self: *const Self) !Self {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return acc * val;
                }
            };
            return try self.reduceAll(scope.op_func, T, null);
        }

        pub fn mean(self: *const Self, dim: usize) !Self {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return acc + val;
                }
                fn post_func(acc: T, count: usize) T {
                    return acc / @as(T, @floatFromInt(count));
                }
            };

            return try self.reduce(dim, scope.op_func, T, scope.post_func);
        }

        pub fn meanAll(self: *const Self) !Tensor(0, .{ .T = T }) {
            const scope = struct {
                fn op_func(acc: T, val: T) T {
                    return acc + val;
                }
                fn post_func(acc: T, count: usize) T {
                    return acc / @as(T, @floatFromInt(count));
                }
            };

            return try self.reduceAll(scope.op_func, T, scope.post_func);
        }

        pub fn anyTrue(self: *const Self, dim: usize) !Tensor(N, .{ .T = bool }) {
            if (T != bool) @compileError("unsupported type " ++ @typeName(T));

            const scope = struct {
                fn op_func(acc: bool, val: bool) bool {
                    return acc or val;
                }
            };

            return try self.reduce(dim, scope.op_func, bool, null);
        }

        pub fn anyTrueAll(self: *const Self) !Tensor(0, .{ .T = bool }) {
            if (T != bool) @compileError("unsupported type " ++ @typeName(T));

            const scope = struct {
                fn op_func(acc: bool, val: bool) bool {
                    return acc or val;
                }
            };

            return try self.reduceAll(scope.op_func, bool, null);
        }

        pub fn allTrue(self: *const Self, dim: usize) !Tensor(N, .{ .T = bool }) {
            if (T != bool) @compileError("unsupported type " ++ @typeName(T));

            const scope = struct {
                fn op_func(acc: bool, val: bool) bool {
                    return acc and val;
                }
            };

            return try self.reduceAll(dim, scope.op_func, bool, null);
        }

        pub fn allTrueAll(self: *const Self) !Tensor(0, .{ .T = bool }) {
            if (T != bool) @compileError("unsupported type " ++ @typeName(T));

            const scope = struct {
                fn op_func(acc: bool, val: bool) bool {
                    return acc and val;
                }
            };

            return try self.reduceAll(scope.op_func, bool, null);
        }

        // create method
        pub fn fromDataImpl(layout_a: Layout, storage_a: Storage, storage_offset_a: usize) !Self {
            return Self{
                .layout = layout_a,
                .storage = storage_a,
                ._storage_offset = storage_offset_a,
            };
        }

        pub fn to(self: Self, comptime NT: type) !Tensor(N, .{ .T = NT }) {
            if (T == NT) {
                return self;
            }

            const layout = Layout.init(self.shape());

            var new_buf = try self.storage.allocator.alloc(NT, self.size());

            var iter = self.shapeIter();

            while (iter.next()) |idx| {
                const new_idx = try utils.indexToFlat(&idx, &self.shape(), &self.stride());
                const self_data = try self.getData(idx);

                switch (@typeInfo(NT)) {
                    .float => switch (@typeInfo(T)) {
                        .float => new_buf[new_idx] = @floatCast(self_data),
                        .int => new_buf[new_idx] = @floatFromInt(self_data),
                        .bool => new_buf[new_idx] = if (self_data) 1.0 else 0.0,
                        inline else => @compileError("Unsupported src type"),
                    },
                    .int => switch (@typeInfo(T)) {
                        .float => new_buf[new_idx] = @intFromFloat(self_data),
                        .int => new_buf[new_idx] = @intCast(self_data),
                        .bool => new_buf[new_idx] = if (self_data) 1 else 0,
                        inline else => @compileError("Unsupported src type"),
                    },
                    .bool => switch (@typeInfo(T)) {
                        .float, .int => new_buf[new_idx] = self_data != 0,
                        .bool => new_buf[new_idx] = self_data,
                        inline else => @compileError("Unsupported src type"),
                    },
                    inline else => @compileError("Unsupported dst type"),
                }
            }

            const storage = try storage_t.Storage(NT, .Cpu).initImpl(self.storage.allocator, new_buf);

            return Tensor(N, .{ .T = NT }).fromDataImpl(layout, storage, 0);
        }

        pub fn clone(self: *const Self) !Self {
            const layout = self.layout.clone();
            const storage = try self.storage.deepCopy();

            return try Self.fromDataImpl(layout, storage, 0);
        }

        pub fn contiguous(self: Self) !Self {
            if (self.layout.isContiguous()) {
                return self;
            }

            std.debug.print("run contiguous action\n", .{});

            const new_buf = try self.storage.allocator.alloc(T, self.size());

            var data_iter = self.shapeIter();

            const layout = Layout.init(self.shape());

            while (data_iter.next()) |idx| {
                const flat_idx = try utils.indexToFlat(&idx, &layout.shape(), &layout.stride());
                new_buf[flat_idx] = try self.getData(idx);
            }

            const storage = try Storage.initImpl(self.storage.allocator, new_buf);

            return Self.fromDataImpl(layout, storage, 0);
        }

        // attributes
        pub fn broadcastTo(self: Self, target_shape: anytype) !Self {
            const new_layout = try self.layout.broadcastTo(N, target_shape);
            std.debug.print("new_layout: {f}\n", .{new_layout});

            const storage = self.storage.shared();

            return try Self.fromDataImpl(new_layout, storage, self._storage_offset);
        }

        pub fn broadcastTo_(self: *Self, target_shape: anytype) !void {
            const new_layout = try self.layout.broadcastTo(N, target_shape);
            std.debug.print("new_layout: {f}\n", .{new_layout});

            self.layout = new_layout;
        }

        pub fn getData(self: *const Self, indices: [N]usize) !T {
            var idx = try utils.indexToFlat(&indices, &self.shape(), &self.stride());
            idx += self._storage_offset;

            return self.storage.dataSlice()[idx];
        }

        pub fn setData(self: *Self, indices: [N]usize, value: T) !void {
            var idx = try utils.indexToFlat(&indices, &self.shape(), &self.stride());
            idx += self._storage_offset;

            self.storage.dataSlice()[idx] = value;
        }

        // layout view method
        pub fn sharedView(self: *const Self) !Self {
            const new_storage = self.storage.shared();

            return Self{
                ._base = self,
                .storage = new_storage,
                .layout = self.layout,
                ._storage_offset = self._storage_offset,
            };
        }

        pub fn transpose_(self: *Self) void {
            if (N != 2) @compileError("only support 2d tensor");
            self.layout = self.layout.transpose(0, 1) catch unreachable;
        }

        pub fn permute_(self: *const Self, perm: [N]usize) !void {
            self.layout = try self.layout.permute(perm);
        }

        pub fn reshape(self: *const Self, new_shapes: anytype) !Tensor(
            utils.array.getArrayShapeComp(@TypeOf(new_shapes))[0],
            .{ .T = T },
        ) {
            const layout = try self.layout.reshape(new_shapes);
            const storage = self.storage.shared();

            return try Tensor(utils.array.getArrayShapeComp(@TypeOf(new_shapes))[0], .{ .T = T }).fromDataImpl(layout, storage, self._storage_offset);
        }

        pub fn unsqueeze(self: *const Self, dim: usize) !Tensor(N + 1, .{ .T = T }) {
            const layout = try self.layout.unsqueeze(dim);
            const storage = self.storage.shared();

            return try Tensor(N + 1, .{ .T = T }).fromDataImpl(
                layout,
                storage,
                self._storage_offset,
            );
        }

        pub fn squeeze(self: *const Self, dim: usize) !Tensor(N - 1, .{ .T = T }) {
            const layout = try self.layout.squeeze(dim);
            const storage = self.storage.shared();

            return try Tensor(N + 1, .{ .T = T }).fromDataImpl(
                layout,
                storage,
                self._storage_offset,
            );
        }

        // core method
        pub fn shapeIter(self: *const Self) ShapeIterator {
            return ShapeIterator.init(self.shape());
        }

        pub fn deinit(self: *const Self) void {
            self.storage.deinit();
        }

        pub fn size(self: *const Self) usize {
            return self.layout.size();
        }

        pub fn ndim(_: *const Self) usize {
            return N;
        }

        pub fn shape(self: *const Self) [N]usize {
            return self.layout.shape();
        }

        pub fn stride(self: *const Self) [N]usize {
            return self.layout.stride();
        }

        pub fn s_allocator(self: *const Self) std.mem.Allocator {
            return self.storage.allocator;
        }

        pub fn isContiguous(self: *const Self) bool {
            return self.layout.isContiguous();
        }

        pub fn equal(self: Self, other: Self) bool {
            if (!self.layout.equal(other.layout)) return false;

            var self_iter = self.shapeIter();

            while (self_iter.next()) |idx| {
                const sv = self.getData(idx) catch unreachable;
                const ov = other.getData(idx) catch unreachable;

                if (sv != ov) return false;
            }

            return true;
        }

        pub fn approxEqual(self: Self, other: Self, relEps: T, absEps: T) bool {
            if (!self.layout.equal(other.layout)) return false;

            const self_data_slice = self.storage.dataSlice();
            const other_data_slice = other.storage.dataSlice();
            return utils.sliceApproxEqual(T, self_data_slice, other_data_slice, relEps, absEps);
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print(
                \\Tensor{{
                \\.DType = {}
                \\.{f}
                \\.{f}
                \\.Data =
            , .{ T, self.storage, self.layout });

            _ = try writer.write("\n");

            const init_index = [_]usize{0} ** N;
            self.fmtRecursive(writer, 0, init_index) catch |err| {
                std.debug.print("meet failure: {}", .{err});
                return std.Io.Writer.Error.WriteFailed;
            };
            _ = try writer.write("\n}");
        }

        fn fmtRecursive(self: *const Self, writer: *std.Io.Writer, depth: usize, indices: [N]usize) anyerror!void {
            const dims = self.ndim();

            if (depth == dims) {
                try writer.print("{}", .{try self.getData(indices)});
            } else if (depth == dims - 1) {
                try self.fmt1dSlice(writer, depth, indices);
            } else {
                try self.fmtNdSlice(writer, depth, indices);
            }
        }

        fn fmtNdSlice(self: *const Self, writer: *std.Io.Writer, depth: usize, base_indices: [N]usize) anyerror!void {
            if (N == 0) {
                return;
            }

            const pad_show_count = 4;

            const current_dim_size = self.shape()[depth];
            const dims = self.ndim();

            _ = try writer.write("[");

            const show_all = current_dim_size <= 2 * pad_show_count;

            const space_handle_fn = struct {
                fn call(idx: usize, depth_a: usize, dims_a: usize, writer_a: *std.Io.Writer) !void {
                    if (idx > 0) {
                        if (depth_a == dims_a - 2) {
                            _ = try writer_a.write("\n ");
                        } else {
                            _ = try writer_a.write("\n\n ");
                        }

                        for (0..depth_a) |_| {
                            _ = try writer_a.write(" ");
                        }
                    }
                }
            }.call;

            if (show_all) {
                for (0..current_dim_size) |i| {
                    try space_handle_fn(i, depth, dims, writer);

                    var indices = base_indices;

                    indices[depth] = i;

                    try self.fmtRecursive(writer, depth + 1, indices);
                }
            } else {
                for (0..pad_show_count) |i| {
                    try space_handle_fn(i, depth, dims, writer);

                    var indices = base_indices;

                    indices[depth] = i;

                    try self.fmtRecursive(writer, depth + 1, indices);
                }

                if (depth == dims - 2) {
                    _ = try writer.write("\n ");

                    for (0..depth) |_| {
                        _ = try writer.write(" ");
                    }

                    _ = try writer.write("...\n ");
                } else {
                    _ = try writer.write("\n ...\n\n ");
                }

                for (0..depth) |_| {
                    _ = try writer.write(" ");
                }

                for (current_dim_size - pad_show_count..current_dim_size) |i| {
                    try space_handle_fn(i, depth, dims, writer);

                    var indices = base_indices;

                    indices[depth] = i;

                    try self.fmtRecursive(writer, depth + 1, indices);
                }
            }

            _ = try writer.write("]");
        }

        fn fmt1dSlice(self: *const Self, writer: *std.Io.Writer, depth: usize, base_indices: [N]usize) anyerror!void {
            if (N == 0) {
                return;
            }

            const pad_show_count = 5;

            const max_items: usize = if (base_indices.len == 0) 10 else 2 * pad_show_count;
            const current_dim_size = self.shape()[depth];

            const line_size = 18;

            _ = try writer.write("[");

            if (current_dim_size <= max_items) {
                for (0..current_dim_size) |i| {
                    if (i > 0) {
                        if (i % line_size == 0) {
                            _ = try writer.write("\n");
                        } else {
                            _ = try writer.write(" ");
                        }
                    }

                    var idx = base_indices;
                    idx[depth] = i;

                    try writer.print("{}", .{try self.getData(idx)});
                }
            } else {
                for (0..pad_show_count) |i| {
                    if (i > 0) {
                        _ = try writer.write(" ");
                    }

                    var idx = base_indices;
                    idx[depth] = i;

                    try writer.print("{}", .{try self.getData(idx)});
                }
                _ = try writer.write(" ... ");

                for (current_dim_size - pad_show_count..current_dim_size) |i| {
                    var idx = base_indices;
                    idx[depth] = i;

                    try writer.print("{}", .{try self.getData(idx)});

                    if (i < current_dim_size - 1) {
                        _ = try writer.write(" ");
                    }
                }
            }

            _ = try writer.write("]");
        }
    };
}

// create factory method
pub fn fromData(comptime N: usize, comptime T: type, allocator: std.mem.Allocator, arr: []T, shape: [N]usize) !Tensor(N, .{ .T = T }) {
    const layout = layout_t.Layout(N).init(shape);
    const Storage = storage_t.Storage(T, .Cpu);

    const storage = try Storage.initImpl(allocator, arr);

    return Tensor(N, .{ .T = T })
        .fromDataImpl(
        layout,
        storage,
        0,
    );
}

pub fn fromScalar(allocator: std.mem.Allocator, value: anytype) !Tensor(
    0,
    .{ .T = utils.numberTypeComp(@TypeOf(value)) },
) {
    const T = utils.numberTypeComp(@TypeOf(value));

    const layout = layout_t.Layout(0).init([0]usize{});
    const Storage = storage_t.Storage(T, .Cpu);

    const storage = try Storage.full(allocator, 1, @as(T, value));

    return Tensor(0, .{ .T = T })
        .fromDataImpl(
        layout,
        storage,
        0,
    );
}

pub fn fromArray(allocator: std.mem.Allocator, arr: anytype) !Tensor(
    utils.array.getArrayShapeComp(@TypeOf(arr)).len,
    .{ .T = utils.array.getArrayItemTypeComp(@TypeOf(arr)) },
) {
    const shape = comptime utils.array.getArrayShapeComp(@TypeOf(arr));
    const N = comptime utils.array.getArrayNDimComp(@TypeOf(arr));
    const T = comptime utils.array.getArrayItemTypeComp(@TypeOf(arr));
    const element_count = comptime utils.array.getArrayElementCountComp(@TypeOf(arr));

    const new_buf = try allocator.alloc(T, element_count);

    const arr_s: []const T = @ptrCast(&arr);
    // array is in stack, must copy to heap
    @memcpy(new_buf, arr_s);

    const layout = layout_t.Layout(shape.len).init(shape);
    const storage = try storage_t.Storage(T, .Cpu).initImpl(allocator, new_buf);

    return Tensor(N, .{ .T = T }).fromDataImpl(layout, storage, 0);
}

pub fn fromArrayList(allocator: std.mem.Allocator, comptime T: type, arr_list: std.ArrayList(T)) !Tensor(
    1,
    .{ .T = T },
) {
    const layout = layout_t.Layout(1).init([1]usize{arr_list.items.len});
    const storage = try storage_t.Storage(T, .Cpu).initImpl(allocator, arr_list.items);

    return Tensor(1, .{ .T = T }).fromDataImpl(layout, storage, 0);
}

pub fn arange(
    allocator: std.mem.Allocator,
    comptime T: type,
    args: struct { start: T = @as(T, 0), step: T = @as(T, 1), end: T },
) !Tensor(1, .{ .T = T }) {
    const storage = try storage_t.Storage(T, .Cpu)
        .arange(allocator, .{
        .start = args.start,
        .step = args.step,
        .end = args.end,
    });
    const layout = layout_t.Layout(1).init([1]usize{storage.len()});

    return Tensor(1, .{ .T = T }).fromDataImpl(layout, storage, 0);
}

pub fn linspace(allocator: std.mem.Allocator, comptime T: type, args: struct {
    start: T,
    end: T,
    steps: usize,
}) !Tensor(1, .{ .T = T }) {
    const storage = try storage_t.Storage(T, .Cpu)
        .linspace(allocator, .{
        .start = args.start,
        .end = args.end,
        .steps = args.steps,
    });
    const layout = layout_t.Layout(1).init([1]usize{storage.len()});

    return Tensor(1, .{ .T = T })
        .fromDataImpl(layout, storage, 0);
}

pub fn full(allocator: std.mem.Allocator, shapes_a: anytype, value: anytype) !Tensor(
    utils.array.getArrayShapeComp(@TypeOf(shapes_a))[0],
    .{ .T = utils.numberTypeComp(@TypeOf(value)) },
) {
    const NDIM = comptime utils.array.getArrayNDimComp(@TypeOf(shapes_a));
    if (NDIM != 1) @compileError("only support 1-d array");

    const T = utils.numberTypeComp(@TypeOf(value));
    const N = comptime shapes_a.len;

    const Layout = layout_t.Layout(N);
    const Storage = storage_t.Storage(T, .Cpu);
    const TensorI = Tensor(N, .{ .T = T });

    const element_count = utils.product(&shapes_a);

    const layout = Layout.init(shapes_a);
    const storage = try Storage.full(allocator, element_count, value);

    return TensorI.fromDataImpl(layout, storage, 0);
}

pub fn fullLike(allocator: std.mem.Allocator, tensor: anytype, value: anytype) !Tensor(
    @TypeOf(tensor).DIMS,
    .{
        .T = utils.numberTypeComp(@TypeOf(value)),
    },
) {
    return try full(allocator, tensor.shape(), value);
}

pub fn zeros(allocator: std.mem.Allocator, shapes_a: anytype) !Tensor(
    utils.array.getArrayShapeComp(@TypeOf(shapes_a))[0],
    .{},
) {
    const NDIM = comptime utils.array.getArrayNDimComp(@TypeOf(shapes_a));
    if (NDIM != 1) @compileError("only support 1-d array");

    const value: f32 = 0;

    return try full(allocator, shapes_a, value);
}

pub fn zerosLike(allocator: std.mem.Allocator, tensor: anytype) !@TypeOf(tensor) {
    return try zeros(allocator, tensor.shapes());
}

pub fn ones(allocator: std.mem.Allocator, shapes_a: anytype) !Tensor(
    utils.array.getArrayShapeComp(@TypeOf(shapes_a)),
    .{},
) {
    const NDIM = comptime utils.array.getArrayNDimComp(@TypeOf(shapes_a));
    if (NDIM != 1) @compileError("only support 1-d array");

    const value: f32 = 1.0;
    return try full(allocator, shapes_a, value);
}

pub fn onesLike(allocator: std.mem.Allocator, tensor: anytype) !@TypeOf(tensor) {
    return try ones(allocator, tensor.shapes());
}

pub fn eye(allocator: std.mem.Allocator, row: usize, column: usize, value: anytype) !Tensor(
    2,
    .{ .T = @TypeOf(value) },
) {
    var tensor = try zeros(allocator, [2]usize{ row, column });

    for (0..@min(row, column)) |i| {
        tensor.setData([2]usize{ i, i }, value);
    }

    return tensor;
}

pub fn rand(allocator: std.mem.Allocator, shapes_a: anytype, low: anytype, high: @TypeOf(low)) !Tensor(
    utils.array.getArrayShapeComp(@TypeOf(shapes_a))[0],
    .{ .T = utils.floatBasicType(@TypeOf(low)) },
) {
    const NDIM = comptime utils.array.getArrayNDimComp(@TypeOf(shapes_a));
    if (NDIM != 1) @compileError("only support 1-d array");

    const N = comptime utils.array.getArrayShapeComp(@TypeOf(shapes_a))[0];
    const T = utils.floatBasicType(@TypeOf(low));

    const layout = layout_t.Layout(N).init(shapes_a);
    const size = layout.size();

    const storage = try storage_t.Storage(T, .Cpu).rand(
        allocator,
        size,
        low,
        high,
    );
    return try Tensor(N, .{ .T = T }).fromDataImpl(
        layout,
        storage,
        0,
    );
}

pub fn randNorm(allocator: std.mem.Allocator, shapes_a: anytype, mean_a: anytype, stddev: @TypeOf(mean_a)) !Tensor(
    utils.array.getArrayShapeComp(@TypeOf(shapes_a))[0],
    .{ .T = utils.floatBasicType(@TypeOf(mean_a)) },
) {
    const NDIM = comptime utils.array.getArrayNDimComp(@TypeOf(shapes_a));
    if (NDIM != 1) @compileError("only support 1-d array");

    const N = comptime utils.array.getArrayShapeComp(@TypeOf(shapes_a))[0];
    const T = utils.floatBasicType(@TypeOf(mean_a));

    const layout = layout_t.Layout(N).init(shapes_a);
    const size = layout.size();

    const storage = try storage_t.Storage(T, .Cpu).randNorm(allocator, size, mean_a, stddev);
    return try Tensor(N, .{ .T = T }).fromDataImpl(layout, storage, 0);
}

pub fn cat(allocator: std.mem.Allocator, tensors: anytype, dim: usize) !utils.getSliceItemType(@TypeOf(tensors)) {
    const TS = utils.getSliceItemType(@TypeOf(tensors));

    var layouts = try allocator.alloc(TS.Layout, tensors.len);
    defer allocator.free(layouts);
    var storages = try allocator.alloc(TS.Storage, tensors.len);
    defer allocator.free(storages);

    for (tensors, 0..) |t, i| {
        layouts[i] = t.layout;
        storages[i] = t.storage;
    }

    const layout = try TS.Layout.cat(layouts, dim);
    const storage = try TS.Storage.cat(allocator, storages);

    return try TS.fromDataImpl(layout, storage, 0);
}

pub fn stack(allocator: std.mem.Allocator, tensors: anytype, dim: usize) !Tensor(
    utils.getSliceItemType(@TypeOf(tensors)).DIMS + 1,
    utils.getSliceItemType(@TypeOf(tensors)).StorageArgs,
) {
    const TS = utils.getSliceItemType(@TypeOf(tensors));

    var layouts = try allocator.alloc(TS.Layout, tensors.len);
    defer allocator.free(layouts);
    var storages = try allocator.alloc(TS.Storage, tensors.len);
    defer allocator.free(storages);

    for (tensors, 0..) |t, i| {
        layouts[i] = t.layout;
        storages[i] = t.storage;
    }

    const layout = try TS.Layout.stack(layouts, dim);
    const storage = try TS.Storage.cat(allocator, storages);

    return try Tensor(TS.DIMS + 1, TS.StorageArgs).fromDataImpl(layout, storage, 0);
}

test "from data directly" {
    const allocator = std.testing.allocator;

    const arr = [2][3][4]f32{
        [3][4]f32{
            [4]f32{ 1.0, 2.0, 3.0, 4.0 },
            [4]f32{ 5.0, 6.0, 7.0, 8.0 },
            [4]f32{ 9.0, 10.0, 11.0, 12.0 },
        },
        [3][4]f32{
            [4]f32{ 13.0, 14.0, 15.0, 16.0 },
            [4]f32{ 17.0, 18.0, 19.0, 20.0 },
            [4]f32{ 21.0, 22.0, 23.0, 24.0 },
        },
    };

    const t1 = try fromArray(allocator, arr);
    defer t1.deinit();
    try std.testing.expectEqual(t1.size(), 2 * 3 * 4);
    try std.testing.expect(if (@TypeOf(t1).T == f32) true else false);

    std.debug.print("t1: {f}\n", .{t1});
}

test "tensor create" {
    const allocator = std.testing.allocator;

    {
        const t1 = try arange(allocator, i32, .{ .start = 0, .step = 2, .end = 10 });
        defer t1.deinit();
        std.debug.print("t1: {f}\n", .{t1});
    }

    {
        const t2 = try linspace(allocator, f32, .{ .start = 7, .end = 30, .steps = 5 });
        defer t2.deinit();
        std.debug.print("t2: {f}\n", .{t2});
    }

    {
        var a = [2]usize{ 10, 13 };
        const t3 = try full(allocator, a, @as(f32, 10.2));
        defer t3.deinit();
        try std.testing.expect(t3.ndim() == 2);
        try std.testing.expectEqualDeep(a, t3.shape());
        a[0] = 11;
        std.debug.print("t3: {f}\n", .{t3});

        const t4 = try fullLike(allocator, t3, 10.2);
        try std.testing.expect(@TypeOf(t4).T == f32);
        try std.testing.expectEqualDeep(t4.shape(), t3.shape());
        defer t4.deinit();
        std.debug.print("t4: {f}\n", .{t4});
    }

    {
        const t5 = try zeros(allocator, [3]usize{ 2, 3, 5 });
        defer t5.deinit();
        std.debug.print("t5: {f}\n", .{t5});
    }

    {
        const t1 = try rand(allocator, [3]usize{ 1, 2, 3 }, 0.0, 2.0);
        defer t1.deinit();
        const t2 = try rand(allocator, [3]usize{ 2, 2, 3 }, 3.0, 7.0);
        defer t2.deinit();
        const t3 = try randNorm(allocator, [3]usize{ 2, 2, 3 }, 0.0, 2.0);
        defer t3.deinit();

        var mean_a: f32 = 0.0;
        var stddev: f32 = 2.0;
        const t4 = try randNorm(allocator, [3]usize{ 2, 2, 3 }, mean_a, stddev);
        defer t4.deinit();

        mean_a = 10.0;
        stddev = 3.0;
        std.debug.print("mean_a: {} stddev: {} t4: {f}\n", .{ mean_a, stddev, t4 });

        const tc = try cat(allocator, &[3]@TypeOf(t1){ t1, t2, t3 }, 0);
        defer tc.deinit();

        try std.testing.expectEqualDeep(tc.shape(), [3]usize{ 5, 2, 3 });
        std.debug.print("tc: {f}\n", .{tc});

        const meet_err =
            if (stack(allocator, &[3]@TypeOf(t1){ t1, t2, t3 }, 0)) |_| false else |_| true;
        try std.testing.expect(meet_err);

        const ts = try stack(allocator, &[2]@TypeOf(t1){ t2, t3 }, 0);
        defer ts.deinit();

        try std.testing.expectEqualDeep(ts.shape(), [4]usize{ 2, 2, 2, 3 });
        std.debug.print("ts: {f}\n", .{ts});
    }
}

test "split" {
    const allocator = std.testing.allocator;

    const t1 = try rand(allocator, [3]usize{ 5, 2, 3 }, 0.0, 1.0);
    defer t1.deinit();

    std.debug.print("t1: {f}\n", .{t1});

    {
        const result = try t1.split(allocator, 3, 0);
        defer allocator.free(result);
        for (result) |t| {
            defer t.deinit();
            std.debug.print("split t: {f}\n", .{t});
        }
    }

    {
        const result = try t1.chunk(allocator, 3, 0);
        defer allocator.free(result);
        for (result) |t| {
            defer t.deinit();
            std.debug.print("chunk t: {f}\n", .{t});
        }
    }

    {
        const result = try t1.unbind(allocator, 0);
        defer allocator.free(result);

        for (result, 0..) |t, i| {
            defer t.deinit();
            try std.testing.expectEqual(t.storage.refCount(), 6 - i);
        }

        for (result) |t| {
            std.debug.print("unbind t: {f} storage refcount: {*}\n", .{ t, &t.storage._ref_count });
        }
    }
}

test "contiguous test" {
    const allocator = std.testing.allocator;

    var t1 = try rand(allocator, [2]usize{ 3, 5 }, 0.0, 5.0);
    defer t1.deinit();

    try std.testing.expect(t1.isContiguous());
    std.debug.print("t1: {f}\n", .{t1});

    t1.transpose_();
    try std.testing.expect(!t1.isContiguous());
    std.debug.print("t1 transpose_: {f}\n", .{t1});

    const t1tc = try t1.contiguous();
    defer t1tc.deinit();

    std.debug.print("t1tc: {f}\n", .{t1tc});
    try std.testing.expect(t1tc.layout.isContiguous());

    try std.testing.expectEqual(try t1.getData([_]usize{ 3, 2 }), try t1tc.getData([_]usize{ 3, 2 }));
}

test "map basic" {
    const allocator = std.testing.allocator;

    var t = try rand(allocator, [3]usize{ 2, 3, 4 }, 0.0, 1.0);
    defer t.deinit();

    const func1 = struct {
        fn call(x: f32, ctx: f32) f32 {
            return x * ctx;
        }
    }.call;
    var t1 = try t.map(f32, 7.0, func1);
    defer t1.deinit();
    std.debug.print("t1: {f}\n", .{t1});

    const func2 = struct {
        fn call(x: f32, ctx: f32) bool {
            return x >= ctx;
        }
    }.call;
    const t2 = try t1.map(bool, 2.0, func2);
    defer t2.deinit();
    std.debug.print("t2: {f}\n", .{t2});

    const func3 = struct {
        fn call(x: f32, ctx: f32) f32 {
            return x * ctx;
        }
    }.call;
    t1.map_(-1.0, func3);
    std.debug.print("t1: {f}\n", .{t1});
}

test "map bool" {
    const allocator = std.testing.allocator;

    const t1 = try randNorm(allocator, [3]usize{ 3, 2, 4 }, 0.0, 1.0);
    defer t1.deinit();

    const t2 = try t1.ge(0.0);
    defer t2.deinit();

    std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
}

test "math op" {
    const allocator = std.testing.allocator;

    {
        var t1 = try arange(allocator, i32, .{ .end = 10 });
        defer t1.deinit();
        t1.add_(12);
        t1.clamp_(10, 15);

        const t2 = try fullLike(allocator, t1, 1.0);
        try std.testing.expectEqualSlices(usize, &t1.shape(), &t2.shape());
        defer t2.deinit();
        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
    }

    {
        var t1 = try arange(allocator, f32, .{ .end = 8.0 });
        defer t1.deinit();

        t1.sin_();
        std.debug.print("t1: {f}\n", .{t1});

        t1.relu_();
        std.debug.print("relu t1: {f}\n", .{t1});

        t1.sigmoid_();
        std.debug.print("sigmoid t1: {f}\n", .{t1});

        t1.sqrt_();
        std.debug.print("sqrt t1: {f}\n", .{t1});
    }
}

test "data type conversion" {
    const allocator = std.testing.allocator;

    const arr = [5]f32{ 1.0, 2.0, 3.0, -2.0, 0.0 };
    const t1 = try fromArray(allocator, arr);
    defer t1.deinit();
    const t2 = try t1.to(i32);
    defer t2.deinit();
    const t3 = try t1.to(bool);
    defer t3.deinit();

    try std.testing.expectEqual(i32, @TypeOf(t2).T);
    try std.testing.expectEqualSlices(usize, &t1.shape(), &t2.shape());
    try std.testing.expectEqual(bool, @TypeOf(t3).T);
    try std.testing.expectEqual(false, try t3.getData([_]usize{4}));
    try std.testing.expectEqual(true, try t3.getData([_]usize{3}));

    std.debug.print("t1: {f} t2: {f} t3: {f}\n", .{ t1, t2, t3 });
}

test "reduce" {
    const allocator = std.testing.allocator;

    const t1 = try rand(allocator, [2]usize{ 3, 5 }, 0.0, 1.0);
    defer t1.deinit();

    {
        const t2 = try t1.sum(0);
        defer t2.deinit();
        const t3 = try t1.sumAll();
        defer t3.deinit();

        const t4 = try t1.max(0);
        defer t4.deinit();

        const t5 = try t1.maxAll();
        defer t5.deinit();

        const t6 = try t1.mean(0);
        defer t6.deinit();
        const t7 = try t1.meanAll();
        defer t7.deinit();

        std.debug.print("t1: {f} t2: {f} t3: {f} t4: {f} t5: {f} t6: {f} t7: {f}\n", .{ t1, t2, t3, t4, t5, t6, t7 });
        // const t3 = try t1.mean(0);
        // std.debug.print("t1: {f} t2: {f} t2 item: {} t3: {f}\n", .{ t1, t2, try t2.scalarItemComp(DataType.f32), t3 });
    }

    {
        const t2 = try t1.gt(0.5);
        defer t2.deinit();

        const any_t = try t2.anyTrue();
        defer any_t.deinit();
        const any_t_all = try t2.allTrueAll();
        defer any_t_all.deinit();
        const all_t = try t2.allTrue();
        defer all_t.deinit();
        const all_t_all = try t2.allTrueAll();
        defer all_t_all.deinit();

        std.debug.print("t2: {f} any_t: {f} all_t: {f} all_t_all: {f}\n", .{ t2, any_t, all_t, all_t_all });
    }
}

test "binary op" {
    const allocator = std.testing.allocator;

    {
        const t1 = try rand(allocator, [_]usize{ 3, 3 }, 10.0, 20.0);
        defer t1.deinit();
        var t2 = try arange(allocator, f32, .{ .end = 9 });
        defer t2.deinit();

        std.debug.print("t2: {f}\n", .{t2});

        const t2r = try t2.reshape([_]usize{ 3, 3 });
        defer t2r.deinit();

        const t3 = try t1.add(t2r);
        defer t3.deinit();

        std.debug.print("t1: {f} t2: {f} t3: {f}\n", .{ t1, t2, t3 });
    }

    {
        const t1 = try arange(allocator, i32, .{ .end = 10 });
        defer t1.deinit();
        const t2 = try t1.add(t1);
        defer t2.deinit();

        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
    }
}

test "masked" {
    const allocator = std.testing.allocator;

    var t1 = try rand(allocator, [_]usize{ 3, 5 }, -1.0, 1.0);
    defer t1.deinit();

    const arr1 = [3][5]bool{ .{ true, false, true, false, true }, .{ false, false, false, true, false }, .{ false, true, false, true, false } };
    const m1 = try fromArray(allocator, arr1);
    defer m1.deinit();

    std.debug.print("t1: {f}\n", .{t1});

    try t1.maskedFill_(m1, 0.0);

    try std.testing.expectEqual(0.0, t1.getData([_]usize{ 0, 0 }));
    try std.testing.expectEqual(0.0, t1.getData([_]usize{ 0, 2 }));
    try std.testing.expectEqual(0.0, t1.getData([_]usize{ 0, 4 }));
    try std.testing.expectEqual(0.0, t1.getData([_]usize{ 1, 3 }));
    try std.testing.expectEqual(0.0, t1.getData([_]usize{ 2, 1 }));
    try std.testing.expectEqual(0.0, t1.getData([_]usize{ 2, 3 }));
    std.debug.print("masked t1: {f}\n", .{t1});
}

test "nan inf" {
    const allocator = std.testing.allocator;

    const arr1 = [5]f32{ 1.0, std.math.inf(f32), std.math.nan(f32), -std.math.inf(f32), -2.3 };

    var t1 = try fromArray(allocator, arr1);
    defer t1.deinit();

    const is_inf = try t1.isInf();
    defer is_inf.deinit();
    try std.testing.expectEqualSlices(
        bool,
        &[_]bool{ false, true, false, true, false },
        is_inf.storage.dataSlice(),
    );
    const is_pos_inf = try t1.isPositiveInf();
    defer is_pos_inf.deinit();
    try std.testing.expectEqualSlices(
        bool,
        &[_]bool{ false, true, false, false, false },
        is_pos_inf.storage.dataSlice(),
    );
    const is_neg_inf = try t1.isNegativeInf();
    defer is_neg_inf.deinit();
    try std.testing.expectEqualSlices(
        bool,
        &[_]bool{ false, false, false, true, false },
        is_neg_inf.storage.dataSlice(),
    );
    const is_nan = try t1.isNan();
    defer is_nan.deinit();
    try std.testing.expectEqualSlices(
        bool,
        &[_]bool{ false, false, true, false, false },
        is_nan.storage.dataSlice(),
    );
    const is_finite = try t1.isFinite();
    defer is_finite.deinit();
    try std.testing.expectEqualSlices(
        bool,
        &[_]bool{ true, false, false, false, true },
        is_finite.storage.dataSlice(),
    );

    std.debug.print("t1: {f} is_inf: {f} is_pos_inf: {f} is_neg_inf: {f} is_nan: {f} is_finite: {f}\n", .{ t1, is_inf, is_pos_inf, is_neg_inf, is_nan, is_finite });

    t1.nanToNum_(.{ .nan = 0.0 });
    try std.testing.expectEqualSlices(
        f32,
        &[_]f32{ 1.0, std.math.inf(f32), 0.0, -std.math.inf(f32), -2.3 },
        t1.storage.dataSlice(),
    );
    std.debug.print("nan_to_num: {f}\n", .{t1});
    t1.nanToNum_(.{ .nan = 0.0, .posinf = 1.0, .neginf = -3.0 });
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 1.0, 0.0, -3.0, -2.3 }, t1.storage.dataSlice());
    std.debug.print("inf_to_num: {f}\n", .{t1});
}

test "softmax" {
    const allocator = std.testing.allocator;

    const t1 = try fromArray(allocator, [3]f32{ 0.3, 2.9, 4.0 });
    defer t1.deinit();
    const t2 = try t1.softmax();
    defer t2.deinit();

    const td = try fromArray(allocator, [3]f32{ 0.01821127, 0.24519181, 0.73659691 });
    defer td.deinit();

    const approx_equal = t2.approxEqual(td, 0.00001, 0.00001);
    try std.testing.expect(approx_equal);
    std.debug.print("t2: {f} td: {f}\n", .{ t2, td });

    const t3 = try fromArray(allocator, [_][10]f32{
        .{ 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 },
        .{ 0.1, 0.15, 0.5, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 },
    });
    defer t3.deinit();

    const t4 = try t3.softmax();
    defer t4.deinit();

    const t4d = try fromArray(allocator, [_][10]f32{
        .{ 0.09832329, 0.09352801, 0.16210771, 0.0889666, 0.09352801, 0.09832329, 0.0889666, 0.09832329, 0.0889666, 0.0889666 },
        .{ 0.09887603, 0.10394551, 0.1475057, 0.08946673, 0.09405379, 0.09887603, 0.08946673, 0.09887603, 0.08946673, 0.08946673 },
    });
    defer t4d.deinit();

    const t4_approx_equal = t4.approxEqual(t4d, 0.00001, 0.00001);
    try std.testing.expect(t4_approx_equal);
}

test "data item" {
    const allocator = std.testing.allocator;

    {
        const t1 = try fromScalar(allocator, 2);
        defer t1.deinit();

        const item = try t1.dataItem();
        try std.testing.expectEqual(item, 2);

        std.debug.print("t1: {f} item: {}\n", .{ t1, item });
    }

    {
        const t1 = try rand(allocator, [3]usize{ 2, 3, 5 }, -5.0, 5.0);
        defer t1.deinit();
        const t2 = try t1.maxAll();
        defer t2.deinit();

        const item = try t2.dataItem();
        try std.testing.expectEqual(@TypeOf(item), utils.numberTypeComp(@TypeOf(5.0)));

        std.debug.print("t2: {f} item: {}\n", .{ t2, item });
    }
}

test "one hot" {
    const allocator = std.testing.allocator;

    {
        const t1 = try arange(allocator, i32, .{ .end = 10 });
        defer t1.deinit();

        const t2 = try t1.oneHot(u8, null);
        defer t2.deinit();

        const td = try fromArray(allocator, [_][10]u8{
            .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            .{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
            .{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
            .{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
            .{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
            .{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
            .{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
            .{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
            .{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
            .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        });
        defer td.deinit();

        const eql_result = t2.equal(td);
        try std.testing.expect(eql_result);

        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
    }

    {
        const t1 = try arange(allocator, i32, .{ .end = 10 });
        defer t1.deinit();

        const t2 = try t1.reshape([2]usize{ 2, 5 });
        defer t2.deinit();

        const t3 = try t2.oneHot(u8, null);
        defer t3.deinit();

        const td = try fromArray(allocator, [_][5][10]u8{
            .{
                .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                .{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
                .{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
                .{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
                .{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
            },
            .{
                .{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                .{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
                .{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
                .{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
                .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
            },
        });
        defer td.deinit();

        const eql_result = t3.equal(td);
        try std.testing.expect(eql_result);

        std.debug.print("t2: {f} t3: {f}\n", .{ t2, t3 });
    }
}

test "pad" {
    const allocator = std.testing.allocator;

    {
        const t1 = try arange(allocator, i32, .{ .end = 5 });
        defer t1.deinit();

        const t2 = try t1.pad(&.{ 2, 3 }, 20);
        defer t2.deinit();

        const td = try fromArray(allocator, [_]i32{ 20, 20, 0, 1, 2, 3, 4, 20, 20, 20 });
        defer td.deinit();

        const eql_result = t2.equal(td);
        try std.testing.expect(eql_result);

        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
    }

    {
        const t1 = try rand(allocator, [2]usize{ 3, 5 }, 0.0, 1.0);
        defer t1.deinit();

        const t2 = try t1.pad(&.{ 2, 3, 1, 3 }, 20);
        defer t2.deinit();

        try std.testing.expectEqual(t2.shape(), [_]usize{ 7, 10 });

        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
    }
}

test "loss func" {
    const allocator = std.testing.allocator;

    {
        const t = try fromArray(allocator, [_]f32{ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
        defer t.deinit();

        const y = try fromArray(allocator, [_]f32{ 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 });
        defer y.deinit();

        const mse_loss = try y.mseLoss(t);
        defer mse_loss.deinit();
        const cross_entropy_loss = try y.crossEntropy(t);
        defer cross_entropy_loss.deinit();

        const mse_v = try mse_loss.dataItem();
        const cross_entropy_v = try cross_entropy_loss.dataItem();
        try std.testing.expectApproxEqAbs(0.0975, mse_v, 0.0001);
        try std.testing.expectApproxEqAbs(0.5108, cross_entropy_v, 0.0001);

        std.debug.print("mse: {f} cross_entropy: {f}\n", .{ mse_loss, cross_entropy_loss });
    }

    {
        const t = try fromArray(allocator, [_][10]f32{
            .{ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        });
        defer t.deinit();
        const y = try fromArray(allocator, [_][10]f32{
            .{ 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 },
            .{ 0.01, 0.15, 0.06, 0.55, 0.03, 0.07, 0.08, 0.01, 0.01, 0.03 },
        });
        defer y.deinit();

        const mse_loss = try y.mseLoss(t);
        defer mse_loss.deinit();
        const cross_entropy_loss = try y.crossEntropy(t);
        defer cross_entropy_loss.deinit();

        const mse_v = try mse_loss.dataItem();
        const cross_entropy_v = try cross_entropy_loss.dataItem();
        try std.testing.expectApproxEqAbs(0.21849997, mse_v, 0.0001);
        try std.testing.expectApproxEqAbs(0.55433106, cross_entropy_v, 0.0001);

        std.debug.print("mse: {f} cross_entropy: {f}\n", .{ mse_loss, cross_entropy_loss });
    }
}
