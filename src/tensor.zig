const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");
const log = @import("./log.zig");

const dtype_o = @import("./dtype.zig");
const DataType = dtype_o.DataType;
const Scalar = dtype_o.Scalar;

const storage_t = @import("./storage.zig");
const Device = storage_t.Device;
const layout_t = @import("./layout.zig");

const shape_expr = @import("shape_expr.zig");
const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;
const parseSpec = shape_expr.parseSpec;

pub fn TensorView(comptime T: type) type {
    return struct {
        owner: *const anyopaque,
        is_contiguous: bool,
        allocator: std.mem.Allocator,
        is_owned: bool,
        shape: []const usize,
        stride: []const usize,
        data: []T,

        pub fn deinit(self: *Self) void {
            if (self.is_owned) {
                self.allocator.free(self.data);
            }
        }

        const Self = @This();

        pub fn resetData(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        pub fn clone(self: *const Self) !Self {
            const new_data = try self.allocator.alloc(T, self.data.len);
            @memcpy(new_data, self.data);

            return Self{
                .owner = self.owner,
                .is_contiguous = self.is_contiguous,
                .allocator = self.allocator,
                .is_owned = true,
                .shape = self.shape,
                .stride = self.shape,
                .data = new_data,
            };
        }

        fn checkElementwiseCondition(self: *const Self, other: *const Self) !void {
            if (self.data.len != other.data.len) {
                std.debug.print("self data len: {} other data len: {}\n", .{ self.data.len, other.data.len });
                return error.ShapeMismatch;
            }

            if (!self.is_contiguous or !other.is_contiguous) {
                return error.NotContiguous;
            }
        }

        pub fn addSubFused_(self: *Self, alpha: T, other: *const Self, beta: T) !void {
            try checkElementwiseCondition(self, other);

            for (self.data, other.data) |*s, o| {
                s.* = alpha * s.* + beta * o;
            }
        }

        pub fn add_(self: *Self, other: *const Self) !void {
            try checkElementwiseCondition(self, other);

            for (self.data, other.data) |*s, o| {
                s.* += o;
            }
        }

        pub fn sub_(self: *Self, other: *const Self) !void {
            try checkElementwiseCondition(self, other);

            for (self.data, other.data) |*s, o| {
                s.* -= o;
            }
        }

        pub fn addScalar_(self: *Self, value: T) void {
            for (self.data) |*s| {
                s.* += value;
            }
        }

        pub fn subScalar_(self: *Self, value: T) void {
            for (self.data) |*s| {
                s.* -= value;
            }
        }

        pub fn mulScalar_(self: *Self, value: T) void {
            for (self.data) |*s| {
                s.* *= value;
            }
        }

        pub fn mul_(self: *Self, other: *const Self) !void {
            try checkElementwiseCondition(self, other);

            for (self.data, other.data) |*s, o| {
                s.* *= o;
            }
        }

        pub fn div_(self: *Self, other: *const Self) !void {
            try checkElementwiseCondition(self, other);

            for (self.data, other.data) |*s, o| {
                s.* /= o;
            }
        }

        pub fn sqrt_(self: *Self) void {
            for (self.data) |*s| {
                s.* = @sqrt(s.*);
            }
        }
    };
}

pub fn BasicOpFuncsGenerator(comptime T: type) type {
    return struct {
        fn eql(a: T, b: T) bool {
            switch (@typeInfo(T)) {
                .array => |ai| return std.mem.eql(ai.child, &a, &b),
                else => return a == b,
            }
        }
        fn lt(a: T, b: T) bool {
            return a < b;
        }
        fn le(a: T, b: T) bool {
            return a <= b;
        }
        fn gt(a: T, b: T) bool {
            return a > b;
        }
        fn ge(a: T, b: T) bool {
            return a >= b;
        }
        fn add(a: T, b: T) T {
            return a + b;
        }
        fn sub(a: T, b: T) T {
            return a - b;
        }
        fn mul(a: T, b: T) T {
            return a * b;
        }
        fn div(a: T, b: T) T {
            return a / b;
        }
    };
}

pub fn ItemGenerator(comptime N: usize, comptime T: type) type {
    return struct {
        index: [N]usize,
        value: T,

        const Self = @This();

        fn arg(acc: Self, _: usize) [N]usize {
            return acc.index;
        }

        fn mean(acc: Self, count: usize) f64 {
            return @as(f64, acc.value) / @as(f64, @floatFromInt(count));
        }

        fn orOp(acc: Self, val: Self) Self {
            if (T != bool)
                @compileError("only support bool, unsupported type " ++ @typeName(T));
            const v_result = acc.value or val.value;

            return .{ .index = acc.index, .value = v_result };
        }

        fn andOp(acc: Self, val: Self) Self {
            if (T != bool)
                @compileError("only support bool, unsupported type " ++ @typeName(T));

            const v_result = acc.value and val.value;

            return .{ .index = acc.index, .value = v_result };
        }

        fn sum(acc: Self, val: Self) Self {
            const v_result = acc.value + val.value;

            return .{ .index = acc.index, .value = v_result };
        }

        fn max(acc: Self, val: Self) Self {
            if (acc.value >= val.value) {
                return .{
                    .index = acc.index,
                    .value = acc.value,
                };
            } else {
                return .{
                    .index = val.index,
                    .value = val.value,
                };
            }
        }

        fn min(acc: Self, val: Self) Self {
            if (acc.value <= val.value) {
                return .{
                    .index = acc.index,
                    .value = acc.value,
                };
            } else {
                return .{
                    .index = val.index,
                    .value = val.value,
                };
            }
        }

        fn prod(acc: Self, val: Self) Self {
            return .{
                .index = acc.index,
                .value = acc.value * val.value,
            };
        }
    };
}

pub fn reduceWithBoolType(
    comptime N: usize,
    self: *const Tensor(N, .{ .T = bool }),
    dm: usize,
    op_func: fn (acc: ItemGenerator(N, i64), x: ItemGenerator(N, i64)) ItemGenerator(N, i64),
    comptime RT: type,
    post_func: ?fn (acc: ItemGenerator(N, i64), count: usize) RT,
) !Tensor(N, .{ .T = RT }) {
    var shape_i = self.shape();
    shape_i[dm] = 1;

    const data_len = utils.product(&shape_i);
    var new_buf = try self.s_allocator().alloc(RT, data_len);

    var shape_i_iter = layout_t.initShapeIterator(shape_i);
    while (shape_i_iter.next()) |idx| {
        const acc = blk: {
            if (self.shape()[dm] == 1) {
                const v_ii = try self.getData(idx);
                break :blk ItemGenerator(N, i64){ .index = idx, .value = @as(i64, @intFromBool(v_ii)) };
            } else {
                var idx_i = idx;
                var idx_i_1 = idx;
                idx_i_1[dm] = 1;

                var acc = op_func(
                    .{ .index = idx_i, .value = @as(i64, @intFromBool(try self.getData(idx_i))) },
                    .{ .index = idx_i_1, .value = @as(i64, @intFromBool(try self.getData(idx_i_1))) },
                );

                for (2..self.shape()[dm]) |k| {
                    idx_i[dm] = k;
                    acc = op_func(
                        acc,
                        .{ .index = idx_i, .value = @as(i64, @intFromBool(try self.getData(idx_i))) },
                    );
                }
                break :blk acc;
            }
        };

        const res = if (post_func) |pf|
            pf(acc, self.shape()[dm])
        else
            @as(RT, acc.value);

        const flat_idx = try utils.indexShapeToFlat(N, shape_i, idx);
        new_buf[flat_idx] = res;
    }

    const layout = layout_t.Layout(N).init(shape_i);
    const storage = try storage_t.Storage(RT, .Cpu).initImpl(self.s_allocator(), new_buf);

    return try Tensor(N, .{ .T = RT }).fromDataImpl(layout, storage, 0);
}

pub fn reduceAllWithBoolType(
    comptime N: usize,
    self: *const Tensor(N, .{ .T = bool }),
    op_func: fn (acc: ItemGenerator(N, i64), x: ItemGenerator(N, i64)) ItemGenerator(N, i64),
    comptime RT: type,
    post_func: ?fn (acc: ItemGenerator(N, i64), count: usize) RT,
) !Tensor(0, .{ .T = RT }) {
    var shape_iter = layout_t.initShapeIterator(self.shape());

    const idx0 = shape_iter.next().?;
    const idx1 = shape_iter.next().?;

    var acc = op_func(
        .{ .index = idx0, .value = @as(i64, @intFromBool(try self.getData(idx0))) },
        .{ .index = idx1, .value = @as(i64, @intFromBool(try self.getData(idx1))) },
    );

    while (shape_iter.next()) |idx| {
        acc = op_func(
            acc,
            .{ .index = idx, .value = @as(i64, @intFromBool(try self.getData(idx))) },
        );
    }

    const new_buf = try self.s_allocator().alloc(RT, 1);

    const layout = layout_t.Layout(0).init([_]usize{});

    if (post_func) |pf| {
        const count = self.size();
        const posted_v = pf(acc, count);

        new_buf[0] = posted_v;
    } else {
        new_buf[0] = @as(RT, acc.value);
    }

    const storage = try storage_t.Storage(RT, .Cpu).initImpl(
        self.s_allocator(),
        new_buf,
    );

    return try Tensor(0, .{ .T = RT }).fromDataImpl(
        layout,
        storage,
        0,
    );
}

pub fn Tensor(comptime SA: []const SizeExpr, comptime TA: type) type {
    return struct {
        const Storage = storage_t.Storage(T);
        const ShapeIterator = layout_t.ShapeIterator(N);
        const Item = ItemGenerator(N, T);
        const BasicOpFuncs = BasicOpFuncsGenerator(T);

        pub const Tag = "Tensor";
        pub const S = SA;
        pub const T = TA;
        // pub const T = storage_args.T;
        pub const N = SA.len;

        const Layout = layout_t.Layout(SA);

        const Self = @This();

        _base: ?*const anyopaque = null,
        storage: Storage,
        layout: Layout,
        _storage_offset: usize = 0,

        // scope method
        // divide
        // fn computeSplitShape(comptime chunk_size: usize, comptime dim: usize) [N]usize {
        //     if (chunk_size == 0) @compileError("chunk_size must be greater than 0");
        //     if (dim >= N) @compileError("dim must be less than N");
        //     if (S[dim] % chunk_size != 0) @compileError("chunk_size must divide dim_len");

        //     var new_shape = utils.array.comptimeSliceToArray(S);
        //     new_shape[dim] = chunk_size;

        //     return new_shape;
        // }
        // pub fn split(self: *const Self, allocator: std.mem.Allocator, comptime chunk_size: usize, comptime dim: usize) ![chunk_size]Tensor(
        //     &computeSplitShape(chunk_size, dim),
        //     T,
        //     .{},
        // ) {
        //     const new_shape = computeSplitShape(chunk_size, dim);
        //     const new_tensor_t = Tensor(&new_shape, T, .{});

        //     const num_splits = S[dim] / chunk_size;
        //     var result = try allocator.alloc(Self, num_splits);

        //     var offset: usize = 0;
        //     for (0..num_splits) |i| {
        //         const chunk_size_i = if ((offset + chunk_size) <= dim_len) chunk_size else (dim_len - offset);

        //         var new_shape = self.shape();
        //         new_shape[dim] = chunk_size_i;

        //         // must use old strides
        //         const new_strides = self.stride();

        //         const layout = Layout.initRaw(new_shape, new_strides);
        //         result[i] = try Self.fromDataImpl(layout, self.storage.shared(), self._storage_offset + offset * self.stride()[dim]);

        //         offset += chunk_size_i;
        //     }

        //     return result;
        // }

        // pub fn chunk(self: *const Self, allocator: std.mem.Allocator, chunk_count: usize, dim: usize) ![]const Self {
        //     if (dim >= self.ndim()) return error.InvalidDim;

        //     const dim_len = self.shape()[dim];
        //     if (chunk_count == 0 or chunk_count > dim_len) return error.InvalidSplit;

        //     const chunk_size_i = (dim_len + chunk_count - 1) / chunk_count;
        //     return try self.split(allocator, chunk_size_i, dim);
        // }

        pub fn dataItem(self: *const Self) !T {
            var data_iter = self.shapeIter();
            const item = data_iter.next();

            if (item) |i| {
                return try self.getData(i);
            } else {
                return error.EmptyTensor;
            }
        }

        // pub fn unbind(self: *const Self, allocator: std.mem.Allocator, dim: usize) ![]const Tensor(N - 1, storage_args) {
        //     if (dim >= self.ndim()) return error.InvalidDim;

        //     const dim_len = self.shape()[dim];

        //     const NT = Tensor(N - 1, storage_args);

        //     var result = try allocator.alloc(NT, dim_len);

        //     var offset: usize = 0;
        //     for (0..dim_len) |idx| {
        //         var new_shape = [_]usize{0} ** (N - 1);
        //         var new_stride = [_]usize{0} ** (N - 1);

        //         {
        //             var i: usize = 0;
        //             var j: usize = 0;

        //             while (j < N) {
        //                 if (j == dim) {
        //                     j += 1;
        //                 } else {
        //                     new_shape[i] = self.shape()[j];
        //                     new_stride[i] = self.stride()[j];

        //                     i += 1;
        //                     j += 1;
        //                 }
        //             }
        //         }

        //         const layout = layout_t.Layout(N - 1).initRaw(new_shape, new_stride);
        //         result[idx] = try NT.fromDataImpl(layout, self.storage.shared(), self._storage_offset + offset * self.stride()[dim]);

        //         offset += 1;
        //     }

        //     return result;
        // }
        fn computeOneHotShape(comptime num_classes_expr: SizeExpr) [N + 1]SizeExpr {
            var shape_i = [_]SizeExpr{num_classes_expr} ** (N + 1);
            for (0..N) |i| {
                shape_i[i] = SA[i];
            }

            return shape_i;
        }

        pub fn oneHot(self: *const Self, comptime TT: type, comptime num_classes_expr: SizeExpr) !Tensor(&computeOneHotShape(num_classes_expr), TT) {
            switch (@typeInfo(T)) {
                .int => {
                    const new_shape_expr = comptime computeOneHotShape(num_classes_expr);
                    var result_tensor = try full(self.s_allocator(), &new_shape_expr, self.layout.shape_env(), @as(TT, 0));

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

        fn computePaddedShape(comptime pads: []const SizeExpr) [N]SizeExpr {
            if (pads.len % 2 != 0) @compileError("pads must be even");

            comptime var new_shape = utils.array.comptimeSliceToArray(SizeExpr, S);

            const pad_len = pads.len / 2;

            inline for (0..pad_len) |i| {
                const orig_value = comptime new_shape[N - 1 - i];
                const pad1 = comptime pads[2 * i];
                const pad2 = comptime pads[2 * i + 1];

                comptime var result = orig_value.add(&pad1);
                result = comptime result.add(&pad2);

                new_shape[N - 1 - i] = result;
            }

            return new_shape;
        }

        pub fn pad(self: *const Self, comptime pads: []const SizeExpr, value: T) !Tensor(&computePaddedShape(pads), T) {
            const new_shape = comptime computePaddedShape(pads);

            var result = try full(self.s_allocator(), &new_shape, self.layout.shape_env(), value);

            var shape_iter = self.shapeIter();
            while (shape_iter.next()) |idx| {
                var dst_idx = idx;

                for (0..pads.len / 2) |i| {
                    const left_add = pads[2 * i];
                    const left_add_size = try left_add.eval(self.layout.shape_env());
                    dst_idx[N - 1 - i] += left_add_size;
                }

                try result.setData(dst_idx, try self.getData(idx));
            }

            return result;
        }

        fn computeIndexSelectShape(comptime dim: usize, comptime dim_expr: SizeExpr) [N]SizeExpr {
            var new_shape = utils.array.comptimeSliceToArray(SizeExpr, SA);
            new_shape[dim] = dim_expr;

            return new_shape;
        }

        pub fn indexSelect(self: *const Self, comptime dim: usize, comptime dim_expr: SizeExpr, indices: []const usize) !Tensor(
            &computeIndexSelectShape(dim, dim_expr),
            T,
        ) {
            if (dim >= N) @compileError("Invalid dimension");

            for (indices) |idx| {
                if (idx >= self.shape()[dim]) {
                    return error.IndexOutOfBounds;
                }
            }

            const new_shape_expr = comptime computeIndexSelectShape(dim, dim_expr);

            var indices_t = try zeros(self.s_allocator(), usize, &new_shape_expr, self.layout.shape_env());
            defer indices_t.deinit();

            var index_iter = indices_t.shapeIter();
            while (index_iter.next()) |idx| {
                try indices_t.setData(idx, indices[idx[dim]]);
            }

            // log.print(@src(), "indices_t: {f}\n", .{indices_t});

            return try self.gather(dim, &new_shape_expr, self.layout.shape_env(), indices_t);
        }

        pub fn gather(
            self: *const Self,
            comptime dim: usize,
            comptime shape_expr_a: []const SizeExpr,
            shape_env: *const ShapeEnv,
            index_tensor: Tensor(shape_expr_a, usize),
        ) !Tensor(shape_expr_a, T) {
            if (dim >= N) @compileError("Invalid dimension");

            const new_buf = try self.s_allocator().alloc(T, index_tensor.size());

            var index_t_iter = index_tensor.shapeIter();
            while (index_t_iter.next()) |idx| {
                const index_d_v = try index_tensor.getData(idx);
                if (index_d_v >= self.shape()[dim]) {
                    return error.IndexOutOfBounds;
                }

                var idx_input = idx;
                idx_input[dim] = try index_tensor.getData(idx);

                const value_input = try self.getData(idx_input);
                const flat_idx_out = try utils.indexShapeToFlat(N, self.shape(), idx);

                new_buf[flat_idx_out] = value_input;
            }

            const layout = try layout_t.Layout(shape_expr_a).init(shape_env);
            const storage = try Storage.initImpl(self.s_allocator(), new_buf);

            return Tensor(shape_expr_a, T).fromDataImpl(layout, storage, 0);
        }

        pub fn scatter_(self: *Self, dim: usize, index_tensor: Tensor(N, .{ .T = usize }), value_src: anytype) !void {
            const VST = @TypeOf(value_src);

            switch (VST) {
                Self, T => {
                    var index_iter = index_tensor.shapeIter();

                    while (index_iter.next()) |idx| {
                        const index_value = try index_tensor.getData(idx);
                        if (index_value >= self.shape()[dim]) {
                            return error.IndexOutOfBounds;
                        }

                        var idx_self = idx;
                        idx_self[dim] = index_value;

                        const value = if (VST == Self) try value_src.getData(idx) else value_src;

                        try self.setData(idx_self, value);
                    }
                },
                else => @compileError("Unsupported type for scatter_ " ++ @typeName(VST)),
            }
        }

        pub fn mseLoss(self: *const Self, other: *const Self) !Tensor(0, .{ .T = T }) {
            var a = try self.clone();
            defer a.deinit();
            try a.sub_(other);
            a.powi_(2);

            var res = try a.sumAll();
            try res.div_(@as(T, 2));

            return res;
        }

        pub fn crossEntropyLogits(self: *const Self, other: *const Self) !Tensor(
            &[_]SizeExpr{SizeExpr.static(1)} ** N,
            T,
        ) {
            switch (@typeInfo(T)) {
                .float => |_| {
                    const batch_size = switch (N) {
                        1 => 1,
                        2 => self.shape()[0],
                        inline else => @compileError("unsuported dimension"),
                    };

                    var logits = try self.clone();
                    defer logits.deinit();

                    const logits_max = try self.max(N - 1);
                    defer logits_max.deinit();

                    const logits_max_b = logits_max.broadcastTo(SA);
                    defer logits_max_b.deinit();

                    logits.sub_(&logits_max_b);

                    var logits_exp = try logits.clone();
                    defer logits_exp.deinit();
                    logits_exp.exp_();

                    var logits_sum = try logits_exp.sum(N - 1);
                    defer logits_sum.deinit();

                    logits_sum.log_();

                    const logits_sum_b = logits_sum.broadcastTo(SA);
                    defer logits_sum_b.deinit();

                    logits.sub_(&logits_sum_b);

                    logits.mul_(other);

                    var loss = try logits.sumAll();
                    loss.divScalar_(-1.0 * @as(T, @floatFromInt(batch_size)));

                    return loss;
                },
                else => @compileError("unsupported type"),
            }
        }

        pub fn crossEntropy(self: *const Self, other: *const Self) !Tensor(&computeReducedAllShapeExpr(), T) {
            switch (@typeInfo(T)) {
                .float => |_| {
                    const batch_size = switch (N) {
                        1 => 1,
                        2 => self.shape()[0],
                        inline else => @compileError("unsuported dimension"),
                    };

                    var a = try self.clone();
                    defer a.deinit();

                    a.log_();
                    a.mulScalar_(-1.0);

                    a.mul_(other);

                    var res = try a.sumAll();
                    res.divScalar_(@as(T, @floatFromInt(batch_size)));

                    return res;
                },
                else => @compileError("unsupported type"),
            }
        }

        // elementwise method
        pub fn map_(self: *Self, ctx: anytype, func: fn (T, utils.comptimeNumberTypeEraseComp(@TypeOf(ctx))) T) void {
            var iter = self.shapeIter();

            while (iter.next()) |idx| {
                self.setData(idx, func(self.getData(idx) catch unreachable, ctx)) catch unreachable;
            }
        }

        pub fn matmul(self: *const Self, other: anytype) !Tensor(
            &.{ SA[0], @TypeOf(other.*).S[1] },
            T,
        ) {
            if (T != f32 and T != f64) {
                @compileError("only support f32 and f64 matmul" ++ " T: " ++ @typeName(T));
            }

            if (comptime !SA[1].equal(@TypeOf(other.*).S[0])) {
                @compileError("matmul shape mismatch: " ++ std.fmt.comptimePrint("lhs: {f} rhs: {f}\n", .{ SA[1], @TypeOf(other.*).S[0] }));
            }

            var lhs = try self.contiguous();
            defer lhs.deinit();
            var rhs = try other.contiguous();
            defer rhs.deinit();

            const m = lhs.shape()[0];
            const k = lhs.shape()[1];
            const n = rhs.shape()[1];

            const a: []const T = @ptrCast(lhs.storage.dataSlice());
            const b: [*c]const T = @ptrCast(rhs.storage.dataSlice());

            const buf = try self.s_allocator().alloc(T, m * n);

            const c: []T = @ptrCast(buf);

            host.matmul(T, @as([*c]const T, @ptrCast(a)), b, @as([*c]T, @ptrCast(c)), m, n, k);

            const layout = try layout_t.Layout(&.{ SA[0], @TypeOf(other.*).S[1] }).init(self.layout.shape_env());
            const storage = try Storage.initImpl(self.s_allocator(), buf);

            return Tensor(&.{ SA[0], @TypeOf(other.*).S[1] }, T).fromDataImpl(layout, storage, 0);
        }

        pub fn map(
            self: *const Self,
            ctx: anytype,
            comptime RT: type,
            func: fn (T, utils.comptimeNumberTypeEraseComp(@TypeOf(ctx))) RT,
        ) !Tensor(SA, RT) {
            const TI = Tensor(SA, RT);

            var new_buf = try self.storage.allocator.alloc(RT, self.size());

            var iter = self.shapeIter();

            var i: usize = 0;
            while (iter.next()) |idx| {
                new_buf[i] = func(self.getData(idx) catch unreachable, ctx);
                i += 1;
            }

            const layout = Layout.init(self.layout.shape_env()) catch unreachable;
            const storage = try storage_t.Storage(RT).initImpl(self.storage.allocator, new_buf);

            return TI.fromDataImpl(layout, storage, 0);
        }

        pub fn eql(self: *const Self, value: *const Self) !Tensor(S, bool) {
            return try self.binaryOp(value, bool, BasicOpFuncs.eql);
        }

        pub fn lt(self: *const Self, value: anytype) !Tensor(N, .{ .T = bool }) {
            const TV = @TypeOf(value);

            switch (TV) {
                @This() => {
                    return try self.binaryOp(value, bool, BasicOpFuncs.lt);
                },
                T => {
                    return try self.map(value, bool, BasicOpFuncs.lt);
                },
                else => @compileError("unsupported lt argument type" ++ " self: " ++ @typeName(@This()) ++ " input: " ++ @typeName(TV)),
            }
        }

        pub fn leScalar(self: *const Self, value: T) !Tensor(
            SA,
            bool,
        ) {
            return try self.map(value, bool, BasicOpFuncs.le);
        }

        pub fn le(self: *const Self, value: *const Self) !Tensor(
            SA,
            bool,
            .{},
        ) {
            return try self.binaryOp(value, bool, BasicOpFuncs.le);
        }

        pub fn gtScalar(self: *const Self, value: T) !Tensor(SA, bool) {
            return try self.map(value, bool, BasicOpFuncs.gt);
        }

        pub fn gt(self: *const Self, value: Self) !Tensor(SA, bool) {
            return try self.binaryOp(value, bool, BasicOpFuncs.gt);
        }

        pub fn geScalar(self: *const Self, value: T) !Tensor(SA, bool) {
            return try self.map(value, bool, BasicOpFuncs.ge);
        }

        pub fn ge(self: *const Self, value: Self) !Tensor(SA, bool) {
            return try self.binaryOp(value, bool, BasicOpFuncs.ge);
        }

        pub fn maskFill_(self: *Self, mask: Tensor(SA, bool), value: T) !void {
            const mask_i = mask;

            var iter = self.shapeIter();

            while (iter.next()) |idx| {
                if (try mask_i.getData(idx)) {
                    try self.setData(idx, value);
                }
            }
        }

        pub fn binaryOp_(self: *Self, b: *const Self, op_func: fn (x: T, y: T) T) void {
            // inplace method: need broadcast to self shape
            var b_i = b.broadcastTo(SA);
            defer b_i.deinit();

            var iter = self.shapeIter();

            while (iter.next()) |idx| {
                const x = self.getData(idx) catch unreachable;
                const y = b_i.getData(idx) catch unreachable;

                self.setData(idx, op_func(x, y)) catch unreachable;
            }
        }

        pub fn binaryOp(
            self: *const Self,
            other: *const Self,
            comptime RT: type,
            op_func: fn (x: T, y: T) RT,
        ) !Tensor(SA, RT) {
            var new_buf = try self.s_allocator().alloc(RT, utils.product(&self.shape()));

            var iter_a = self.shapeIter();

            var i: usize = 0;

            while (iter_a.next()) |idx| {
                const x = try self.getData(idx);
                const y = try other.getData(idx);

                const flat_idx = try utils.indexShapeToFlat(N, self.shape(), idx);

                new_buf[flat_idx] = op_func(x, y);
                i += 1;
            }

            const layout = Layout.init(self.layout.shape_env()) catch unreachable;
            const storage = try storage_t.Storage(RT).initImpl(
                self.s_allocator(),
                new_buf,
            );

            return Tensor(SA, RT).fromDataImpl(layout, storage, 0);
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

        pub fn addScalar_(self: *Self, value: T) !void {
            const vv = @as(T, value);

            return self.map_(vv, BasicOpFuncs.add);
        }

        pub fn add_(self: *Self, value: *const Self) void {
            return self.binaryOp_(value, BasicOpFuncs.add);
        }

        pub fn add(self: *const Self, value: anytype) !Tensor(N, .{ .T = if (T == bool) isize else T }) {
            const TV = @TypeOf(value);
            switch (TV) {
                @This() => {
                    const RT = if (T == bool) isize else T;
                    const func = struct {
                        fn call(v: T, other: T) RT {
                            if (T == bool) {
                                const v1: RT = @intFromBool(v);
                                const v2: RT = @intFromBool(other);
                                return v1 + v2;
                            }
                            return v + other;
                        }
                    }.call;

                    return try self.binaryOp(@as(@This(), value), RT, func);
                },
                T => {
                    return try self.map(value, T, BasicOpFuncs.add);
                },
                else => @compileError("unsupported add argument type" ++ " self: " ++ @typeName(@This()) ++ " input: " ++ @typeName(TV)),
            }
        }
        pub fn subScalar_(self: *Self, value: T) void {
            return self.map_(value, BasicOpFuncs.sub);
        }

        pub fn sub_(self: *Self, value: *const Self) void {
            return self.binaryOp_(value, BasicOpFuncs.sub);
        }

        pub fn sub(self: *const Self, value: *const Self) !Self {
            return self.binaryOp(value, T, BasicOpFuncs.sub);
        }

        pub fn mulScalar_(self: *Self, value: T) void {
            return self.map_(value, BasicOpFuncs.mul);
        }

        pub fn mul_(self: *Self, value: *const Self) void {
            return self.binaryOp_(value, BasicOpFuncs.mul);
        }

        pub fn mul(self: *const Self, value: *const Self) !Self {
            return try self.binaryOp(value, T, BasicOpFuncs.mul);
        }

        pub fn divScalar_(self: *Self, value: T) void {
            return self.map_(value, BasicOpFuncs.div);
        }

        pub fn div_(self: *Self, value: *const Self) !void {
            return self.binaryOp_(value, BasicOpFuncs.div);
        }

        pub fn div(self: *const Self, value: *const Self) !Self {
            return try self.binaryOp(value, T, BasicOpFuncs.div);
        }

        pub fn divScalar(self: *const Self, value: anytype) !Tensor(S, @TypeOf(value)) {
            const DT = @TypeOf(value);
            const func = struct {
                fn call(v: T, ctx: DT) DT {
                    const vt = utils.promoteNumberType(DT, v);

                    return vt / ctx;
                }
            }.call;
            return try self.map(value, DT, func);
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
                    // const ce = 1e-7;
                    // if (v < ce) {
                    //     return @log(v + ce);
                    // }
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

        pub fn abs_(self: *Self) void {
            const func = struct {
                fn call(v: T, _: void) T {
                    return @abs(v);
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
            return try self.map(void{}, bool, func);
        }

        pub fn isInf(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isInf(v);
                }
            }.call;
            return try self.map(void{}, bool, func);
        }

        pub fn isPositiveInf(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isPositiveInf(v);
                }
            }.call;
            return try self.map(void{}, bool, func);
        }

        pub fn isNegativeInf(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isNegativeInf(v);
                }
            }.call;
            return try self.map(void{}, bool, func);
        }

        pub fn isFinite(self: *const Self) !Tensor(N, .{ .T = bool }) {
            const func = struct {
                fn call(v: T, _: void) bool {
                    return std.math.isFinite(v);
                }
            }.call;
            return try self.map(void{}, bool, func);
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

            const a_b = a.broadcastTo(SA);
            defer a_b.deinit();

            var v = try self.sub(&a_b);
            v.exp_();

            const vs = try v.sum(N - 1);
            defer vs.deinit();

            const vs_b = vs.broadcastTo(SA);
            defer vs_b.deinit();

            try v.div_(&vs_b);

            return v;
        }

        // reduce method
        fn computeReducedShapeExpr(comptime dm: usize) [N]SizeExpr {
            var shape_i = [_]SizeExpr{undefined} ** N;
            inline for (0..N) |i| {
                shape_i[i] = if (i == dm) SizeExpr.static(1) else SA[i];
            }
            return shape_i;
        }
        fn computeReducedShape(shape_a: [N]usize, comptime dm: usize) [N]usize {
            var shape_i = [_]usize{undefined} ** N;
            inline for (0..N) |i| {
                shape_i[i] = if (i == dm) 1 else shape_a[i];
            }
            return shape_i;
        }

        fn computeReducedAllShapeExpr() [N]SizeExpr {
            return [_]SizeExpr{SizeExpr.static(1)} ** N;
        }

        pub fn reduce(
            self: *const Self,
            comptime dm: usize,
            op_func: fn (acc: Item, x: Item) Item,
            comptime RT: type,
            post_func: ?fn (acc: Item, count: usize) RT,
        ) !Tensor(&computeReducedShapeExpr(dm), RT) {
            const shape_expr_i = comptime computeReducedShapeExpr(dm);
            const shape_i = computeReducedShape(self.shape(), dm);

            const data_len = utils.product(&shape_i);

            var new_buf = try self.s_allocator().alloc(RT, data_len);

            var shape_i_iter = ShapeIterator.init(shape_i);

            while (shape_i_iter.next()) |idx| {
                const acc = blk: {
                    if (self.shape()[dm] == 1) {
                        const v_ii = try self.getData(idx);
                        break :blk Item{ .index = idx, .value = v_ii };
                    } else {
                        var idx_i = idx;
                        var idx_i_1 = idx;
                        idx_i_1[dm] = 1;

                        var acc = op_func(
                            .{ .index = idx_i, .value = try self.getData(idx_i) },
                            .{ .index = idx_i_1, .value = try self.getData(idx_i_1) },
                        );

                        for (2..self.shape()[dm]) |k| {
                            idx_i[dm] = k;
                            acc = op_func(
                                acc,
                                .{ .index = idx_i, .value = try self.getData(idx_i) },
                            );
                        }
                        break :blk acc;
                    }
                };

                const res = if (post_func) |pf|
                    pf(acc, self.shape()[dm])
                else
                    @as(RT, acc.value);

                const flat_idx = try utils.indexShapeToFlat(N, shape_i, idx);
                new_buf[flat_idx] = res;
            }

            const layout = layout_t.Layout(&shape_expr_i).init(self.layout.shape_env()) catch unreachable;
            const storage = try storage_t.Storage(RT).initImpl(self.s_allocator(), new_buf);

            return Tensor(&shape_expr_i, RT).fromDataImpl(layout, storage, 0);
        }

        pub fn reduceAll(
            self: *const Self,
            op_func: fn (acc: Item, x: Item) Item,
            comptime RT: type,
            post_func: ?fn (acc: Item, count: usize) RT,
        ) !Tensor(&computeReducedAllShapeExpr(), RT) {
            var shape_iter = ShapeIterator.init(self.shape());

            const idx0 = shape_iter.next().?;
            const idx1 = shape_iter.next().?;

            var acc = op_func(
                .{ .index = idx0, .value = try self.getData(idx0) },
                .{ .index = idx1, .value = try self.getData(idx1) },
            );

            while (shape_iter.next()) |idx| {
                acc = op_func(
                    acc,
                    .{ .index = idx, .value = try self.getData(idx) },
                );
            }

            const new_shape_expr = comptime computeReducedAllShapeExpr();
            const new_buf = try self.s_allocator().alloc(RT, 1);

            const layout = layout_t.Layout(&new_shape_expr).init(self.layout.shape_env()) catch unreachable;

            if (post_func) |pf| {
                const count = self.size();
                const posted_v = pf(acc, count);

                new_buf[0] = posted_v;
            } else {
                new_buf[0] = @as(RT, acc.value);
            }

            const storage = try storage_t.Storage(RT).initImpl(
                self.s_allocator(),
                new_buf,
            );

            return Tensor(&new_shape_expr, RT).fromDataImpl(
                layout,
                storage,
                0,
            );
        }

        pub fn sum(self: *const Self, comptime dim: usize) !Tensor(
            &computeReducedShapeExpr(dim),
            if (T == bool) i64 else T,
        ) {
            if (T == bool) {
                return try reduceWithBoolType(N, self, dim, ItemGenerator(N, i64).sum, i64, null);
            } else {
                return try self.reduce(dim, Item.sum, T, null);
            }
        }

        pub fn sumAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), if (T == bool) i64 else T) {
            if (T == bool) {
                const tmp = try self.to(i64);
                defer tmp.deinit();

                return try tmp.sumAll();
            } else {
                return try self.reduceAll(Item.sum, T, null);
            }
        }

        pub fn max(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), T) {
            return try self.reduce(dim, Item.max, T, null);
        }

        pub fn argMax(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), [N]usize) {
            return try self.reduce(dim, Item.max, [N]usize, Item.arg);
        }

        pub fn maxAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), T) {
            return try self.reduceAll(Item.max, T, null);
        }

        pub fn argMaxAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), [N]usize) {
            return try self.reduceAll(Item.max, [N]usize, Item.arg);
        }

        pub fn min(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), T) {
            return try self.reduce(dim, Item.min, T, null);
        }

        pub fn argMin(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), [N]usize) {
            return try self.reduce(dim, Item.min, [N]usize, Item.arg);
        }

        pub fn minAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), T, .{}) {
            return try self.reduceAll(Item.min, T, null);
        }

        pub fn argMinAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), [N]usize, .{}) {
            return try self.reduceAll(Item.min, [N]usize, Item.arg);
        }

        pub fn prod(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), T, .{}) {
            return try self.reduce(dim, Item.prod, T, null);
        }

        pub fn prodAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), T, .{}) {
            return try self.reduceAll(Item.prod, T, null);
        }

        pub fn mean(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), f64) {
            return try self.reduce(dim, Item.sum, f64, Item.mean);
        }

        pub fn meanAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), f64) {
            return try self.reduceAll(Item.sum, f64, Item.mean);
        }

        pub fn anyTrue(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), bool) {
            return try self.reduce(dim, Item.orOp, bool, null);
        }

        pub fn anyTrueAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), bool) {
            return try self.reduceAll(Item.orOp, bool, null);
        }

        pub fn allTrue(self: *const Self, comptime dim: usize) !Tensor(&computeReducedShapeExpr(dim), bool) {
            return try self.reduce(dim, Item.andOp, bool, null);
        }

        pub fn allTrueAll(self: *const Self) !Tensor(&computeReducedAllShapeExpr(), bool) {
            return try self.reduceAll(Item.andOp, bool, null);
        }

        // create method
        pub fn fromDataImpl(layout_a: Layout, storage_a: Storage, storage_offset_a: usize) Self {
            return Self{
                .layout = layout_a,
                .storage = storage_a,
                ._storage_offset = storage_offset_a,
            };
        }

        pub fn to(self: *const Self, comptime NT: type) !Tensor(SA, NT) {
            const layout = Layout.init(self.layout.shape_env()) catch unreachable;

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

            const storage = try storage_t.Storage(NT).initImpl(self.storage.allocator, new_buf);

            return Tensor(SA, NT).fromDataImpl(layout, storage, 0);
        }

        pub fn clone(self: *const Self) !Self {
            const layout = self.layout;
            const storage = try self.storage.deepCopy();

            return Self.fromDataImpl(layout, storage, 0);
        }

        pub fn contiguous(self: *const Self) !Self {
            if (self.layout.isContiguous()) {
                return self.sliceView(.{});
            }

            // std.debug.print("run contiguous action\n", .{});

            const new_buf = try self.storage.allocator.alloc(T, self.size());

            var data_iter = self.shapeIter();

            const layout = try Layout.init(self.layout.shape_env());

            while (data_iter.next()) |idx| {
                const flat_idx = try utils.indexToFlat(&idx, &layout.shape(), &layout.stride());
                new_buf[flat_idx] = try self.getData(idx);
            }

            const storage = try Storage.initImpl(self.storage.allocator, new_buf);

            return Tensor(S, T).fromDataImpl(layout, storage, self._storage_offset);
        }

        // attributes
        pub fn broadcastTo(self: *const Self, comptime target_shape_expr: []const SizeExpr) Tensor(
            target_shape_expr,
            T,
        ) {
            const new_layout = self.layout.broadcastTo(target_shape_expr);

            const storage = self.storage.shared();

            return Tensor(target_shape_expr, T).fromDataImpl(new_layout, storage, self._storage_offset);
        }

        pub fn view(self: *const Self) TensorView(T) {
            return TensorView(T){
                .owner = self,
                .is_contiguous = self.isContiguous(),
                .is_owned = false,
                .allocator = self.s_allocator(),
                .shape = self.shapeRef(),
                .stride = self.strideRef(),
                .data = self.dataSliceRaw(),
            };
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
        fn computeSharedViewShapeErasedPrefixSize(comptime slice_views: anytype) usize {
            switch (@typeInfo(@TypeOf(slice_views))) {
                .@"struct" => |si| {
                    comptime var erased_size: usize = 0;

                    inline for (si.fields) |field| {
                        switch (@typeInfo(field.type)) {
                            .int, .comptime_int => erased_size += 1,
                            else => break,
                        }
                    }

                    return erased_size;
                },
                else => @compileError("Unsupported type for computeSharedViewShapeLen"),
            }
        }

        const ComputeRangeOffset = struct {
            shape: [N]usize,
            offset: usize,
        };

        const Range = struct {
            start: usize,
            end: usize,
        };

        fn computeSliceViewRanges(comptime slice_views: anytype) [utils.stt.getFieldsLenComptime(@TypeOf(slice_views))]Range {
            switch (@typeInfo(@TypeOf(slice_views))) {
                .@"struct" => |si| {
                    const FN = si.fields.len;

                    comptime var ranges = [_]Range{undefined} ** FN;

                    inline for (0..FN) |i| {
                        const range = switch (@typeInfo(si.fields[i].type)) {
                            .int, .comptime_int => blk: {
                                const v = slice_views[i];

                                switch (SA[i]) {
                                    .Static => |static_info| {
                                        if (v >= static_info) {
                                            @compileLog("Invalid slice view: " ++ std.fmt.comptimePrint("SA[{d}] = {d}, slice_views[{d}] = {d}\n", .{ i, SA[i], i, v }));
                                            @compileError("Slice view out of bounds");
                                        }
                                    },
                                    else => {},
                                }

                                break :blk Range{ .start = v, .end = v + 1 };
                            },
                            .@"struct" => blk: {
                                const ran = @as(struct { usize, usize }, slice_views[i]);
                                const start = ran.@"0";
                                const end = ran.@"1";

                                if (start > end) {
                                    @compileError("Invalid slice view: " ++ std.fmt.comptimePrint("SA[{d}] = {d}, slice_views[{d}] = {any}\n", .{ i, SA[i], i, ran }));
                                }

                                switch (SA[i]) {
                                    .Static => |static_info| {
                                        if ((start >= static_info) or (end > static_info)) {
                                            @compileError("Invalid slice view: " ++ std.fmt.comptimePrint("SA[{d}] = {f}, slice_views[{d}] = {any}\n", .{ i, SA[i], i, ran }));
                                        }
                                    },
                                    else => {},
                                }

                                break :blk Range{ .start = start, .end = end };
                            },
                            else => @compileError("Unsupported type for slice_views"),
                        };

                        ranges[i] = range;
                    }
                    return ranges;
                },
                else => @compileError("Unsupported type for slice_views"),
            }
        }

        fn computeSharedViewBaseOffset(self: *const Self, comptime slice_views: anytype) usize {
            const ranges = computeSliceViewRanges(slice_views);

            var base_idx = [_]usize{0} ** N;
            inline for (ranges, 0..) |range, i| {
                base_idx[i] = range.start;
            }

            const base_offset = utils.indexShapeToFlat(N, self.shape(), base_idx) catch unreachable;
            return base_offset;
        }

        fn computeSharedViewShape(comptime slice_views: anytype) [N - computeSharedViewShapeErasedPrefixSize(slice_views)]SizeExpr {
            const ranges = computeSliceViewRanges(slice_views);
            comptime var base_shape = utils.array.comptimeSliceToArray(SizeExpr, SA);

            inline for (ranges, 0..) |range, i| {
                base_shape[i] = SizeExpr.static(range.end - range.start);
            }

            // @compileLog("base shape: " ++ std.fmt.comptimePrint("{any}\n", .{base_shape}));

            const erased_prefix = comptime computeSharedViewShapeErasedPrefixSize(slice_views);

            comptime var shape_i = [_]SizeExpr{SizeExpr.static(0)} ** (N - erased_prefix);

            inline for (erased_prefix..N) |i| {
                shape_i[i - erased_prefix] = base_shape[i];
            }

            // @compileLog("shape_i: " ++ std.fmt.comptimePrint("{any}\n", .{shape_i}));

            return shape_i;
        }

        pub fn sliceView(self: *const Self, comptime slice_views: anytype) Tensor(
            &computeSharedViewShape(slice_views),
            T,
        ) {
            const new_shape_expr = comptime computeSharedViewShape(slice_views);
            const base_offset = computeSharedViewBaseOffset(self, slice_views);

            const erased_prefix = comptime computeSharedViewShapeErasedPrefixSize(slice_views);

            var new_stride = [_]usize{0} ** new_shape_expr.len;

            inline for (erased_prefix..N) |i| {
                new_stride[i - erased_prefix] = self.stride()[i];
            }

            return Tensor(&new_shape_expr, T){
                ._base = self,
                .storage = self.storage.shared(),
                .layout = layout_t.Layout(&new_shape_expr).initRaw(self.layout.shape_env(), new_stride) catch unreachable,
                ._storage_offset = self._storage_offset + base_offset,
            };
        }

        pub fn transpose(self: *const Self) Tensor(&layout_t.computePermutedShapeExpr(SA, &.{ 1, 0 }), T) {
            return self.permute([_]usize{ 1, 0 });
        }

        pub fn permute(self: *const Self, comptime perm: [N]usize) Tensor(&layout_t.computePermutedShapeExpr(SA, &perm), T) {
            const new_layout = self.layout.permute(perm);

            // std.debug.print("permute: old= {f} new= {f}\n", .{ self.layout, new_layout });
            const new_storage = self.storage.shared();

            return Tensor(&layout_t.computePermutedShapeExpr(SA, &perm), T){
                ._base = self,
                .storage = new_storage,
                .layout = new_layout,
                ._storage_offset = self._storage_offset,
            };
        }

        pub fn reshape(self: *const Self, comptime new_shape_expr: []const SizeExpr) Tensor(
            new_shape_expr,
            T,
        ) {
            const layout = self.layout.reshape(new_shape_expr);
            const storage = self.storage.shared();

            return Tensor(new_shape_expr, T).fromDataImpl(layout, storage, self._storage_offset);
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

            return try Tensor(N - 1, .{ .T = T }).fromDataImpl(
                layout,
                storage,
                self._storage_offset,
            );
        }

        // core method
        pub fn shapeIter(self: *const Self) ShapeIterator {
            return ShapeIterator.init(self.shape());
        }

        pub fn dataSliceRaw(self: *const Self) []T {
            return self.storage.dataSlice()[self._storage_offset .. self._storage_offset + self.size()];
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

        pub fn shapeRef(self: *const Self) []const usize {
            return &self.layout.shape();
        }

        pub fn stride(self: *const Self) [N]usize {
            return self.layout.stride();
        }

        pub fn strideRef(self: *const Self) []const usize {
            return &self.layout.stride();
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

                switch (@typeInfo(T)) {
                    .array => |ai| return std.mem.eql(ai.child, &sv, &ov),
                    else => return sv == ov,
                }
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

        pub fn typeInfo() []const u8 {
            comptime var info: []const u8 = @typeName(T) ++ ": ";
            inline for (S) |se| {
                info = info ++ "[" ++ std.fmt.comptimePrint("{f}", .{se}) ++ "]";
            }

            return info;
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print(
                \\Tensor{{
                \\.DType = {}
                \\.{f}
                \\.BaseOffset = {}
                \\.{f}
                \\.Data =
            , .{ T, self.storage, self._storage_offset, self.layout });

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
                try writer.print("{any}", .{try self.getData(indices)});
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

                    try writer.print("{any}", .{try self.getData(idx)});
                }
            } else {
                for (0..pad_show_count) |i| {
                    if (i > 0) {
                        _ = try writer.write(" ");
                    }

                    var idx = base_indices;
                    idx[depth] = i;

                    try writer.print("{any}", .{try self.getData(idx)});
                }
                _ = try writer.write(" ... ");

                for (current_dim_size - pad_show_count..current_dim_size) |i| {
                    var idx = base_indices;
                    idx[depth] = i;

                    try writer.print("{any}", .{try self.getData(idx)});

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
pub fn fromData(
    comptime T: type,
    allocator: std.mem.Allocator,
    arr: []T,
    comptime shape_expr_a: []const SizeExpr,
    shape_env: *const ShapeEnv,
) !Tensor(shape_expr_a, T) {
    const layout = try layout_t.Layout(shape_expr_a).init(shape_env);
    const Storage = storage_t.Storage(T);

    const storage = try Storage.initImpl(allocator, arr);

    return Tensor(shape_expr_a, T)
        .fromDataImpl(
        layout,
        storage,
        0,
    );
}

pub fn fromScalar(allocator: std.mem.Allocator, value: anytype) !Tensor(
    0,
    .{ .T = utils.comptimeNumberTypeEraseComp(@TypeOf(value)) },
) {
    const T = utils.comptimeNumberTypeEraseComp(@TypeOf(value));

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

pub fn fromArraySpecifyDimension(allocator: std.mem.Allocator, comptime D: usize, arr: anytype) !Tensor(
    &shape_expr.staticShapeExpr(&utils.array.getArrayShapeCompWithDepth(@TypeOf(arr), D)),
    utils.array.getArrayItemTypeCompWithDepth(@TypeOf(arr), D),
) {
    const shape = comptime utils.array.getArrayShapeCompWithDepth(@TypeOf(arr), D);

    const shape_expr_i = comptime shape_expr.staticShapeExpr(&shape);

    const AN = comptime utils.array.getArrayNDimComp(@TypeOf(arr));
    if (D > AN) {
        @compileError("can't specify dimension larger than arr, arr dimension: " ++ D);
    }

    const T = comptime utils.array.getArrayItemTypeCompWithDepth(@TypeOf(arr), D);
    const element_count = comptime utils.array.getArrayElementCountCompWithDepth(@TypeOf(arr), D);

    const new_buf = try allocator.alloc(T, element_count);

    const arr_s: []const T = @ptrCast(&arr);
    // array is in stack, must copy to heap
    @memcpy(new_buf, arr_s);

    const layout = try layout_t.Layout(&shape_expr_i).init(&ShapeEnv.init(allocator));
    const storage = try storage_t.Storage(T).initImpl(allocator, new_buf);

    return Tensor(&shape_expr_i, T).fromDataImpl(layout, storage, 0);
}

pub fn fromArray(allocator: std.mem.Allocator, arr: anytype, shape_env: *const ShapeEnv) !Tensor(
    &utils.array.getArrayShapeComp(@TypeOf(arr)),
    utils.array.getArrayItemTypeComp(@TypeOf(arr)),
) {
    const shape_expr_i = comptime utils.array.getArrayShapeComp(@TypeOf(arr));
    const T = comptime utils.array.getArrayItemTypeComp(@TypeOf(arr));

    const element_count = comptime utils.array.getArrayElementCountComp(@TypeOf(arr));

    const new_buf = try allocator.alloc(T, comptime element_count.static_value());

    const arr_s: []const T = @ptrCast(&arr);
    // array is in stack, must copy to heap
    @memcpy(new_buf, arr_s);

    const layout = try layout_t.Layout(&shape_expr_i).init(shape_env);
    const storage = try storage_t.Storage(T).initImpl(allocator, new_buf);

    return Tensor(&shape_expr_i, T).fromDataImpl(layout, storage, 0);
}

pub fn fromArrayList(allocator: std.mem.Allocator, comptime T: type, arr_list: std.ArrayList(T)) !Tensor(
    1,
    T,
) {
    const layout = layout_t.Layout(1).init([1]usize{arr_list.items.len});
    const storage = try storage_t.Storage(T, .Cpu).initImpl(allocator, arr_list.items);

    return Tensor(1, .{ .T = T }).fromDataImpl(layout, storage, 0);
}

fn computeArangeCount(start: anytype, step: @TypeOf(start), end: @TypeOf(start)) usize {
    var count: usize = 0;

    var tmp = start;
    while (tmp < end) {
        count += 1;
        tmp += step;
    }

    return count;
}

pub fn arange(
    allocator: std.mem.Allocator,
    end: anytype,
    comptime len_sym: shape_expr.SymbolHandle,
    shape_env: *ShapeEnv,
    comptime args: struct {
        const T = utils.comptimeNumberTypeEraseComp(@TypeOf(end));
        start: T = @as(T, 0),
        step: T = @as(T, 1),
    },
) !Tensor(
    &.{SizeExpr{ .Sym = len_sym }},
    utils.comptimeNumberTypeEraseComp(@TypeOf(end)),
) {
    const T = utils.comptimeNumberTypeEraseComp(@TypeOf(end));

    const element_count = computeArangeCount(args.start, args.step, end);
    try shape_env.bind(&len_sym, element_count);

    const storage = try storage_t.Storage(T)
        .arange(allocator, .{
        .start = args.start,
        .step = args.step,
        .end = end,
    });

    const shape_expr_i = &.{SizeExpr{ .Sym = len_sym }};
    const layout = try layout_t.Layout(shape_expr_i).init(shape_env);

    return Tensor(shape_expr_i, T).fromDataImpl(layout, storage, 0);
}

pub fn linspace(
    allocator: std.mem.Allocator,
    start: anytype,
    end: utils.comptimeNumberTypeEraseComp(@TypeOf(start)),
    comptime steps: usize,
    comptime args: struct { shape: ?[]const usize = null },
) !Tensor(&shape_expr.staticShapeExpr(&.{steps}), utils.comptimeNumberTypeEraseComp(@TypeOf(start))) {
    const T = utils.comptimeNumberTypeEraseComp(@TypeOf(start));

    const shape = if (args.shape) |s| blk: {
        const s_count = comptime utils.product(s);
        if (s_count != steps) @compileError("shape does not match steps");
        break :blk s;
    } else &.{steps};

    const shape_expr_i = comptime shape_expr.staticShapeExpr(shape);

    const storage = try storage_t.Storage(T)
        .linspace(allocator, start, end, steps);
    const layout = try layout_t.Layout(&shape_expr_i).init(&ShapeEnv.init(allocator));

    return Tensor(&shape_expr_i, T)
        .fromDataImpl(layout, storage, 0);
}

pub fn full(allocator: std.mem.Allocator, comptime shape_expr_a: []const SizeExpr, shape_env: *const ShapeEnv, value: anytype) !Tensor(
    shape_expr_a,
    utils.comptimeNumberTypeEraseComp(@TypeOf(value)),
) {
    const T = utils.comptimeNumberTypeEraseComp(@TypeOf(value));

    const Layout = layout_t.Layout(shape_expr_a);
    const Storage = storage_t.Storage(T);
    const TensorI = Tensor(shape_expr_a, T);

    const layout = try Layout.init(shape_env);
    const storage = try Storage.full(allocator, layout.size(), value);

    return TensorI.fromDataImpl(layout, storage, 0);
}

pub fn fullLike(allocator: std.mem.Allocator, tensor: anytype, value: anytype) !Tensor(
    @TypeOf(tensor).S,
    utils.comptimeNumberTypeEraseComp(@TypeOf(value)),
) {
    return try full(allocator, @TypeOf(tensor).S, tensor.layout.shape_env(), value);
}

pub fn zeros(allocator: std.mem.Allocator, comptime T: type, comptime shape_expr_a: []const SizeExpr, shape_env: *const ShapeEnv) !Tensor(
    shape_expr_a,
    T,
) {
    const value: T = 0;

    return try full(allocator, shape_expr_a, shape_env, value);
}

pub fn zerosLike(allocator: std.mem.Allocator, tensor: anytype) !@TypeOf(tensor) {
    return try zeros(allocator, @TypeOf(tensor).T, @TypeOf(tensor).S, tensor.layout.shape_env());
}

pub fn ones(allocator: std.mem.Allocator, shapes_a: anytype) !Tensor(
    utils.array.getArrayShapeComp(@TypeOf(shapes_a)),
) {
    const NDIM = comptime utils.array.getArrayNDimComp(@TypeOf(shapes_a));
    if (NDIM != 1) @compileError("only support 1-d array");

    const value: f32 = 1.0;
    return try full(allocator, shapes_a, value);
}

pub fn onesLike(allocator: std.mem.Allocator, tensor: anytype) !@TypeOf(tensor) {
    return try ones(allocator, tensor.shapes());
}

pub fn eye(allocator: std.mem.Allocator, row: usize, column: usize, value: anytype) !Tensor(2, @TypeOf(value)) {
    var tensor = try zeros(allocator, [2]usize{ row, column });

    for (0..@min(row, column)) |i| {
        tensor.setData([2]usize{ i, i }, value);
    }

    return tensor;
}

pub fn rand(allocator: std.mem.Allocator, comptime shape_expr_a: []const SizeExpr, shape_env: *const ShapeEnv, low: anytype, high: @TypeOf(low)) !Tensor(
    shape_expr_a,
    utils.comptimeNumberTypeEraseComp(@TypeOf(low)),
) {
    const T = utils.comptimeNumberTypeEraseComp(@TypeOf(low));

    const layout = try layout_t.Layout(shape_expr_a).init(shape_env);
    const size = layout.size();

    const storage = try storage_t.Storage(T).rand(
        allocator,
        size,
        low,
        high,
    );
    return Tensor(shape_expr_a, T).fromDataImpl(
        layout,
        storage,
        0,
    );
}

pub fn randNorm(allocator: std.mem.Allocator, comptime shape_expr_a: []const SizeExpr, shape_env: *const ShapeEnv, mean_a: anytype, stddev: @TypeOf(mean_a)) !Tensor(
    shape_expr_a,
    utils.floatBasicType(@TypeOf(mean_a)),
) {
    const T = utils.floatBasicType(@TypeOf(mean_a));

    const layout = try layout_t.Layout(shape_expr_a).init(shape_env);
    const size = layout.size();

    const storage = try storage_t.Storage(T).randNorm(allocator, size, mean_a, stddev);
    return Tensor(shape_expr_a, T).fromDataImpl(layout, storage, 0);
}

// pub fn cat(allocator: std.mem.Allocator, tensors: anytype, comptime dim: usize) !Tensor(
//     &utils.tensor.computeCattedTensorShape(@TypeOf(tensors), dim),
//     utils.tensor.computeTensorsElementType(@TypeOf(tensors)),
//     .{},
// ) {
//     const shape = comptime utils.tensor.computeCattedTensorShape(@TypeOf(tensors), dim);
//     const T = utils.tensor.computeTensorsElementType(@TypeOf(tensors));

//     const layout = layout_t.Layout(&shape).init();
//     const storage = try storage_t.Storage(T, .Cpu).cat(
//         allocator,
//         &utils.tensor.computeTensorsStorages(tensors),
//     );

//     return Tensor(&shape, T, .{}).fromDataImpl(layout, storage, 0);
// }

// pub fn stack(allocator: std.mem.Allocator, tensors: anytype, comptime dim: usize) !Tensor(
//     &utils.tensor.computeStackedTensorShape(@TypeOf(tensors), dim),
//     utils.tensor.computeTensorsElementType(@TypeOf(tensors)),
//     .{},
// ) {
//     const shape = comptime utils.tensor.computeStackedTensorShape(@TypeOf(tensors), dim);
//     const T = utils.tensor.computeTensorsElementType(@TypeOf(tensors));

//     const layout = layout_t.Layout(&shape).init();
//     const storage = try storage_t.Storage(T, .Cpu).cat(
//         allocator,
//         &utils.tensor.computeTensorsStorages(tensors),
//     );

//     return Tensor(&shape, T, .{}).fromDataImpl(layout, storage, 0);
// }

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
        const t1 = try arange(allocator, 10, .{ .start = 0, .step = 2 });
        defer t1.deinit();
        std.debug.print("t1: {f}\n", .{t1});
    }

    {
        const t2 = try linspace(allocator, 7.0, 30, 5, .{});
        defer t2.deinit();
        std.debug.print("t2: {f}\n", .{t2});
    }

    var shape_env = ShapeEnv.init(allocator);
    defer shape_env.deinit();
    {
        const a = comptime [_]SizeExpr{ SizeExpr.static(10), SizeExpr.static(13) };
        const t3 = try full(allocator, &a, &shape_env, @as(f32, 10.2));
        defer t3.deinit();
        try std.testing.expect(t3.ndim() == 2);
        try std.testing.expectEqualDeep([2]usize{ 10, 13 }, t3.shape());
        // a[0] = 11;
        // std.debug.print("t3: {f}\n", .{t3});

        const t4 = try fullLike(allocator, t3, @as(f32, 10.2));
        try std.testing.expect(@TypeOf(t4).T == f32);
        try std.testing.expectEqualDeep(t4.shape(), t3.shape());
        defer t4.deinit();
        std.debug.print("t4: {f}\n", .{t4});
    }

    {
        const t5 = try zeros(allocator, f32, &parseSpec(.{ 2, 3, 5 }), &shape_env);
        defer t5.deinit();
        std.debug.print("t5: {f}\n", .{t5});
    }

    {
        const t1 = try rand(allocator, &parseSpec(.{ 1, 2, 3 }), &shape_env, 0.0, 2.0);
        defer t1.deinit();
        const t2 = try rand(allocator, &parseSpec(.{ 2, 2, 3 }), &shape_env, 3.0, 7.0);
        defer t2.deinit();
        const t3 = try randNorm(allocator, &parseSpec(.{ 2, 2, 3 }), &shape_env, 0.0, 2.0);
        defer t3.deinit();

        var mean_a: f32 = 0.0;
        var stddev: f32 = 2.0;
        const t4 = try randNorm(allocator, &parseSpec(.{ 2, 2, 3 }), &shape_env, mean_a, stddev);
        defer t4.deinit();

        mean_a = 10.0;
        stddev = 3.0;
        std.debug.print("mean_a: {} stddev: {} t4: {f}\n", .{ mean_a, stddev, t4 });

        // const tc = try cat(allocator, .{ t1, t2, t3 }, 0);
        // defer tc.deinit();

        // try std.testing.expectEqualDeep(tc.shape(), [3]usize{ 5, 2, 3 });
        // std.debug.print("tc: {f}\n", .{tc});

        // const ts = try stack(allocator, .{ t2, t3 }, 0);
        // defer ts.deinit();

        // try std.testing.expectEqualDeep(ts.shape(), [4]usize{ 2, 2, 2, 3 });
        // std.debug.print("ts: {f}\n", .{ts});
    }
}

// test "split" {
//     const allocator = std.testing.allocator;

//     const t1 = try rand(allocator, &.{ 5, 2, 3 }, 0.0, 1.0);
//     defer t1.deinit();

//     std.debug.print("t1: {f}\n", .{t1});

//     {
//         const result = try t1.split(allocator, 3, 0);
//         defer allocator.free(result);
//         for (result) |t| {
//             defer t.deinit();
//             std.debug.print("split t: {f}\n", .{t});
//         }
//     }

//     {
//         const result = try t1.chunk(allocator, 3, 0);
//         defer allocator.free(result);
//         for (result) |t| {
//             defer t.deinit();
//             std.debug.print("chunk t: {f}\n", .{t});
//         }
//     }

//     {
//         const result = try t1.unbind(allocator, 0);
//         defer allocator.free(result);

//         for (result, 0..) |t, i| {
//             defer t.deinit();
//             try std.testing.expectEqual(t.storage.refCount(), 6 - i);
//         }

//         for (result) |t| {
//             std.debug.print("unbind t: {f} storage refcount: {*}\n", .{ t, &t.storage._ref_count });
//         }
//     }
// }

test "matmul" {
    const allocator = std.testing.allocator;

    var shape_env = ShapeEnv.init(allocator);
    defer shape_env.deinit();

    var t1 = try rand(allocator, &.{ SizeExpr.static(3), SizeExpr.static(5) }, &shape_env, 0.0, 5.0);
    defer t1.deinit();

    var t2 = try rand(allocator, &.{ SizeExpr.static(5), SizeExpr.static(2) }, &shape_env, 0.0, 5.0);
    defer t2.deinit();

    var result = try t1.matmul(&t2);
    defer result.deinit();

    std.debug.print("matmul result: {f}\n", .{result});
}

test "contiguous test" {
    const allocator = std.testing.allocator;

    var shape_env = ShapeEnv.init(allocator);
    defer shape_env.deinit();

    var t1 = try rand(allocator, &.{ SizeExpr.static(3), SizeExpr.static(5) }, &shape_env, 0.0, 5.0);
    defer t1.deinit();

    try std.testing.expect(t1.isContiguous());
    std.debug.print("t1: {f}\n", .{t1});

    var t2 = t1.transpose();
    defer t2.deinit();

    try std.testing.expect(!t2.isContiguous());
    std.debug.print("t1 transpose_: {f}\n", .{t1});

    var t1tc = try t1.contiguous();
    defer t1tc.deinit();

    std.debug.print("t1tc: {f}\n", .{t1tc});
    try std.testing.expect(t1tc.layout.isContiguous());

    try std.testing.expectEqual(try t2.getData([_]usize{ 3, 2 }), try t1.getData([_]usize{ 2, 3 }));
}

test "map basic" {
    const allocator = std.testing.allocator;

    var t = try rand(allocator, &.{ 2, 3, 4 }, 0.0, 1.0);
    defer t.deinit();

    const func1 = struct {
        fn call(x: f64, ctx: f64) f64 {
            return x * ctx;
        }
    }.call;
    var t1 = try t.map(7.0, f64, func1);
    defer t1.deinit();
    std.debug.print("t1: {f}\n", .{t1});

    const func2 = struct {
        fn call(x: f64, ctx: f64) bool {
            return x >= ctx;
        }
    }.call;
    const t2 = try t1.map(2.0, bool, func2);
    defer t2.deinit();
    std.debug.print("t2: {f}\n", .{t2});

    const func3 = struct {
        fn call(x: f64, ctx: f64) f64 {
            return x * ctx;
        }
    }.call;
    t1.map_(-1.0, func3);
    std.debug.print("t1: {f}\n", .{t1});
}

test "map bool" {
    const allocator = std.testing.allocator;

    const t1 = try randNorm(allocator, &.{ 3, 2, 4 }, 0.0, 1.0);
    defer t1.deinit();

    const t2 = try t1.geScalar(0.0);
    defer t2.deinit();

    std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
}

test "math op" {
    const allocator = std.testing.allocator;

    {
        var t1 = try arange(allocator, 10, .{});
        defer t1.deinit();
        try t1.addScalar_(12);
        t1.clamp_(10, 15);

        const t2 = try fullLike(allocator, t1, 1.0);
        try std.testing.expectEqualSlices(usize, &t1.shape(), &t2.shape());
        defer t2.deinit();
        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
    }

    {
        var t1 = try arange(allocator, 8.0, .{});
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

    var shape_env = ShapeEnv.init(allocator);
    defer shape_env.deinit();

    const t1 = try rand(allocator, &parseSpec(.{ 3, 5 }), &shape_env, 0.0, 1.0);
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
        const t2 = try t1.gtScalar(0.5);
        defer t2.deinit();

        const any_t = try t2.anyTrue(0);
        defer any_t.deinit();
        const any_t_all = try t2.anyTrueAll();
        defer any_t_all.deinit();
        const all_t = try t2.allTrue(0);
        defer all_t.deinit();
        const all_t_all = try t2.allTrueAll();
        defer all_t_all.deinit();

        std.debug.print("t2: {f} any_t: {f} any_t_all: {f} all_t: {f} all_t_all: {f}\n", .{ t2, any_t, any_t_all, all_t, all_t_all });
    }
}

test "reduce arg" {
    const allocator = std.testing.allocator;

    const t2 = try fromArray(allocator, [2][3]f32{
        .{ 0.7, 0.2, 0.3 },
        .{ 0.4, 0.5, 0.6 },
    });
    defer t2.deinit();

    {
        const r1 = try t2.max(0);
        defer r1.deinit();

        const r1_e = try fromArray(allocator, [_][3]f32{
            .{ 0.7, 0.5, 0.6 },
        });
        defer r1_e.deinit();
        try std.testing.expect(r1.equal(r1_e));

        const r2 = try t2.argMax(0);
        defer r2.deinit();

        const r2_e = try fromArraySpecifyDimension(allocator, 2, [1][3][2]usize{
            .{
                .{ 0, 0 },
                .{ 1, 1 },
                .{ 1, 2 },
            },
        });
        defer r2_e.deinit();

        log.print(@src(), "r2: {f} r2_e: {f}\n", .{ r2, r2_e });
        try std.testing.expect(r2.equal(r2_e));

        const r3 = try t2.maxAll();
        defer r3.deinit();
        const r4 = try t2.argMaxAll();
        defer r4.deinit();

        try std.testing.expectEqual(try r4.dataItem(), [2]usize{ 0, 0 });
        log.print(@src(), "r1: {f} r2: {f} r3: {f} r4: {f}\n", .{ r1, r2, r3, r4 });
    }
}

test "binary op" {
    const allocator = std.testing.allocator;

    {
        const t1 = try fromArray(allocator, [3]bool{ true, false, true });
        defer t1.deinit();
        const t2 = try fromArray(allocator, [3]bool{ false, true, true });
        defer t2.deinit();
        const t3 = try t1.add(t2);
        defer t3.deinit();
        std.debug.print("t1: {f} t2: {f} t3: {f}\n", .{ t1, t2, t3 });
    }

    {
        const t1 = try rand(allocator, [_]usize{ 3, 3 }, 10.0, 20.0);
        defer t1.deinit();
        var t2 = try arange(allocator, f64, .{ .end = 9 });
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

    try t1.maskFill_(m1, 0.0);

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

    {
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

    {
        const t1 = try fromArray(allocator, [_][3]f64{
            .{ 1.05414809, 0.63071653, 1.1328074 },
        });
        defer t1.deinit();
        const t2 = try t1.softmax();
        defer t2.deinit();

        const expected_t2 = try fromArray(allocator, [_][3]f64{.{ 0.36541271, 0.23927078, 0.39531651 }});
        defer expected_t2.deinit();

        const t2_approx_equal = t2.approxEqual(expected_t2, 1e-8, 1e-8);
        try std.testing.expect(t2_approx_equal);
    }
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
        try std.testing.expectEqual(@TypeOf(item), utils.comptimeNumberTypeEraseComp(@TypeOf(5.0)));

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

        const y_c = try y.clone();
        defer y_c.deinit();

        const mse_loss = try y_c.mseLoss(&t);
        defer mse_loss.deinit();
        const cross_entropy_loss = try y.crossEntropy(&t);
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

        const y_c = try y.clone();
        defer y_c.deinit();

        const mse_loss = try y_c.mseLoss(&t);
        defer mse_loss.deinit();
        const cross_entropy_loss = try y.crossEntropy(&t);
        defer cross_entropy_loss.deinit();

        const mse_v = try mse_loss.dataItem();
        const cross_entropy_v = try cross_entropy_loss.dataItem();
        try std.testing.expectApproxEqAbs(0.21849997, mse_v, 0.0001);
        try std.testing.expectApproxEqAbs(0.55433106, cross_entropy_v, 0.0001);

        std.debug.print("mse: {f} cross_entropy: {f}\n", .{ mse_loss, cross_entropy_loss });
    }

    {
        const y = try fromArray(allocator, [3]f64{ 3.0, 4.0, 5.0 });
        defer y.deinit();
        const t = try fromArray(allocator, [3]f64{ 0.0, 0.0, 1.0 });
        defer t.deinit();

        const ce_logits = blk: {
            const ce = try y.crossEntropyLogits(&t);
            defer ce.deinit();
            const cross_entropy = try ce.dataItem();
            break :blk cross_entropy;
        };

        const softmax_ce = blk: {
            const y_s = try y.softmax();
            defer y_s.deinit();
            const cross_entropy_t = try y_s.crossEntropy(&t);
            defer cross_entropy_t.deinit();

            const cross_entropy = try cross_entropy_t.dataItem();
            break :blk cross_entropy;
        };

        try std.testing.expectApproxEqAbs(0.4076059644443804, ce_logits, 1e-8);
        try std.testing.expectApproxEqAbs(0.4076059644443804, softmax_ce, 1e-8);

        // std.debug.print("ce: logits= {} softmax= {}\n", .{ ce_logits, softmax_ce });

        // try std.testing.expectApproxEqAbs(0.9280682857864075, cross_entropy, 1e-7);
    }
}

test "gather_scatter_indexed" {
    const allocator = std.testing.allocator;

    {
        const t1 = try fromArray(allocator, [2][2]i32{
            .{ 1, 2 },
            .{ 3, 4 },
        });
        defer t1.deinit();

        const index_t = try fromArray(allocator, [2][2]usize{
            .{ 0, 0 },
            .{ 1, 0 },
        });
        defer index_t.deinit();

        const t2 = try t1.gather(1, index_t);
        defer t2.deinit();

        const t2_e = try fromArray(allocator, [2][2]i32{
            .{ 1, 1 },
            .{ 4, 3 },
        });
        defer t2_e.deinit();

        const compare_result = t2.equal(t2_e);
        try std.testing.expect(compare_result);
    }

    {
        const t1 = try fromArray(allocator, [3][4]f32{
            .{ 1.0, 2.0, 3.0, 4.0 },
            .{ 5.0, 6.0, 7.0, 8.0 },
            .{ 9.0, 10.0, 11.0, 12.0 },
        });
        defer t1.deinit();
        log.print(@src(), "indexSelct: t1= {f}\n", .{t1});

        {
            const t2 = try t1.indexSelect(1, &.{ 1, 2 });
            defer t2.deinit();

            const t2_e = try fromArray(allocator, [3][2]f32{
                .{ 2.0, 3.0 },
                .{ 6.0, 7.0 },
                .{ 10.0, 11.0 },
            });
            defer t2_e.deinit();

            const compare_result = t2.equal(t2_e);
            try std.testing.expect(compare_result);

            std.debug.print("indexSelect: t2= {f}\n", .{t2});
        }

        {
            const t2 = try t1.indexSelect(0, &.{ 1, 2 });
            defer t2.deinit();

            const t2_e = try fromArray(allocator, [2][4]f32{
                .{ 5.0, 6.0, 7.0, 8.0 },
                .{ 9.0, 10.0, 11.0, 12.0 },
            });
            defer t2_e.deinit();

            const compare_result = t2.equal(t2_e);
            try std.testing.expect(compare_result);

            std.debug.print("indexSelect: t2= {f}\n", .{t2});
        }
    }

    {
        const t1 = try arange(allocator, i8, .{ .start = 1, .end = 11 });
        defer t1.deinit();
        const t2 = try t1.reshape([2]usize{ 2, 5 });
        defer t2.deinit();

        const index_t = try fromArray(allocator, [2][2]usize{
            .{ 0, 1 },
            .{ 2, 0 },
        });
        defer index_t.deinit();

        var input = try zeros(allocator, i8, [2]usize{ 3, 5 });
        defer input.deinit();

        try input.scatter_(0, index_t, t2);

        const t3_e = try fromArray(allocator, [3][5]i8{ .{ 1, 0, 0, 4, 0 }, .{ 0, 2, 0, 0, 0 }, .{ 0, 0, 3, 0, 0 } });
        defer t3_e.deinit();

        const compare_result = input.equal(t3_e);
        try std.testing.expect(compare_result);
    }

    {
        var input = try zeros(allocator, f32, [2]usize{ 3, 5 });
        defer input.deinit();

        const index_t = try fromArray(allocator, [1][2]usize{.{ 0, 1 }});
        defer index_t.deinit();

        try input.scatter_(0, index_t, @as(f32, 2.0));

        const input_e = try fromArray(
            allocator,
            [3][5]f32{
                .{ 2.0, 0.0, 0.0, 0.0, 0.0 },
                .{ 0.0, 2.0, 0.0, 0.0, 0.0 },
                .{ 0.0, 0.0, 0.0, 0.0, 0.0 },
            },
        );
        defer input_e.deinit();

        const compare_result = input.equal(input_e);
        try std.testing.expect(compare_result);

        std.debug.print("input: {f}\n", .{input});
    }
}

test "shared_view" {
    const allocator = std.testing.allocator;

    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    var input = try rand(allocator, &.{ SizeExpr.static(3), SizeExpr.static(5) }, &shape_env, 2.0, 5.0);
    defer input.deinit();
    std.debug.print("input: {f}\n", .{input});

    var iv1 = input.sliceView(.{});
    defer iv1.deinit();
    try std.testing.expectEqual(iv1.shape(), [2]usize{ 3, 5 });

    var iv2 = input.sliceView(.{1});
    defer iv2.deinit();
    try std.testing.expectEqual(iv2.shape(), [1]usize{5});

    var iv3 = input.sliceView(.{ 2, 4 });
    defer iv3.deinit();
    try std.testing.expectEqual(iv3.shape(), [0]usize{});
    std.debug.print("iv3: {f}\n", .{iv3});

    var iv4 = input.sliceView(.{ .{ 1, 3 }, .{ 1, 5 } });
    defer iv4.deinit();
    std.debug.print("iv4: {f}\n", .{iv4});

    // try std.testing.expectEqual(iv4.shape(), [2]usize{ 2, 2 });
    // try iv4.setData([2]usize{ 0, 1 }, @as(f32, 3.0));
    // try iv4.setData([2]usize{ 1, 0 }, @as(f32, 4.0));

    // try std.testing.expectEqual(@as(f32, 3.0), try input.getData([2]usize{ 1, 2 }));
    // try std.testing.expectEqual(@as(f32, 4.0), try input.getData([2]usize{ 2, 1 }));

    // std.debug.print("input: {f} iv1= {f} iv2= {f} iv3= {f} iv4= {f}\n", .{ input, iv1, iv2, iv3, iv4 });
}
