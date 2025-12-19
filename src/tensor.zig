const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");

const dtype_o = @import("./dtype.zig");
const DataType = dtype_o.DataType;
const Scalar = dtype_o.Scalar;

const storage_t = @import("./storage.zig");
const Device = storage_t.Device;
const layout_t = @import("./layout.zig");

pub fn Tensor(comptime N: usize, comptime storage_args: struct {
    T: type = f32,
    comptime D: Device = .Cpu,
}) type {
    // const ShapeIterator = layout_t.ShapeIterator(N);

    return struct {
        const StorageArgs = storage_args;
        const Storage = storage_t.Storage(storage_args.T, storage_args.D);

        const T = storage_args.T;
        const D = storage_args.D;
        const Layout = layout_t.Layout(N);

        const DIMS = N;

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

        // // elementwise method
        // pub fn map_(self: *Self, comptime data_type: DataType, ctx: anytype, func: fn (data_type.toTypeComp(), @TypeOf(ctx)) data_type.toTypeComp()) !void {
        //     var iter = try self.dataIter();

        //     while (iter.next()) |idx| {
        //         const x = try self.getWithIndicesCompType(data_type, idx);
        //         x.* = func(x.*, ctx);
        //     }
        // }

        // pub fn map(self: *const Self, comptime data_type: DataType, comptime return_type: DataType, ctx: anytype, func: fn (data_type.toTypeComp(), @TypeOf(ctx)) return_type.toTypeComp()) !Self {
        //     var new_buf = try self.allocator.alloc(return_type.toTypeComp(), self.layout.size());

        //     var iter = try self.dataIter();

        //     var i: usize = 0;
        //     while (iter.next()) |idx| {
        //         const x = try self.getWithIndicesCompType(data_type, idx);
        //         new_buf[i] = func(x.*, ctx);
        //         i += 1;
        //     }

        //     const layout = try Layout.init(self.allocator, return_type, self.shapes());
        //     const storage = Storage.init(self.allocator, Storage.Device.Cpu, @ptrCast(new_buf.ptr), new_buf.len * return_type.dtypeSize());

        //     return try Self.fromDataImpl(self.allocator, layout, storage, 0);
        // }

        // pub fn mapBool(self: *const Self, ctx: anytype, func: fn (T, @TypeOf(ctx)) bool) !Tensor(bool) {
        //     var new_buf = try self.allocator.alloc(bool, self.layout.size());

        //     var iter = try self.dataIter();
        //     defer iter.deinit();

        //     var tmp: usize = 0;
        //     while (iter.next()) |idx| {
        //         const v = try self.get(idx);
        //         new_buf[tmp] = func(v, ctx);

        //         tmp += 1;
        //     }

        //     const layout = try Layout.init(self.allocator, DataType.bool, self.shapes());
        //     const storage = Storage.init(self.allocator, Storage.Device.Cpu, @ptrCast(new_buf.ptr), new_buf.len * @sizeOf(bool));

        //     return Self.fromDataImpl(self.allocator, layout, storage, self._storage_offset);
        // }

        // pub fn eql(self: *const Self, value: T) !Tensor(bool) {
        //     const scope = struct {
        //         fn call(v: T, ctx: T) bool {
        //             return v == ctx;
        //         }
        //     }.call;
        //     return try self.mapBool(value, scope);
        // }

        // pub fn lt(self: *const Self, value: T) !Tensor(bool) {
        //     const scope = struct {
        //         fn call(v: T, ctx: T) bool {
        //             return v < ctx;
        //         }
        //     }.call;
        //     return try self.mapBool(value, scope);
        // }

        // pub fn gt(self: *const Self, value: T) !Tensor(bool) {
        //     const scope = struct {
        //         fn call(v: T, ctx: T) bool {
        //             return v > ctx;
        //         }
        //     }.call;
        //     return try self.mapBool(value, scope);
        // }

        // pub fn maskedFill_(self: *Self, mask: Tensor(bool), value: T) !void {
        //     const a = try mask.broadcastTo(self.shapes());

        //     var iter = try self.dataIter();
        //     defer iter.deinit();

        //     while (iter.next()) |idx| {
        //         if ((try a.getWithIndicesCompType(DataType.bool, idx)).*) {
        //             switch (self.dtype()) {
        //                 inline else => |dt| {
        //                     const v = dtype_o.toDType(dt.toTypeComp(), value);
        //                     (try self.getWithIndicesCompType(dt, idx)).* = v;
        //                 },
        //             }
        //         }
        //     }
        // }

        // pub fn binaryOp_(self: *Self, b: Self, comptime data_type: DataType, op_func: fn (x: data_type.toTypeComp(), y: data_type.toTypeComp()) data_type.toTypeComp()) !void {
        //     // inplace method: need broadcast to self shape
        //     var b_i = b;
        //     try b_i.broadcastTo_(self.shapes());

        //     var iter = try self.dataIter();
        //     defer iter.deinit();

        //     while (iter.next()) |idx| {
        //         const x = try self.getWithIndicesCompType(data_type, idx);
        //         const y = try b_i.getWithIndicesCompType(data_type, idx);

        //         x.* = op_func(x.*, y.*);
        //     }
        // }

        // pub fn binaryOp(
        //     self: *const Self,
        //     b: Self,
        //     comptime data_type: DataType,
        //     op_func: fn (x: data_type.toTypeComp(), y: data_type.toTypeComp()) data_type.toTypeComp(),
        // ) !Self {
        //     const target_shapes = try utils.compatibleBroacastShapes(
        //         self.allocator,
        //         self.shapes(),
        //         b.shapes(),
        //     );

        //     const a = try self.broadcastTo(target_shapes.items);
        //     const c = try b.broadcastTo(target_shapes.items);

        //     var new_buf = try self.allocator.alloc(data_type.toTypeComp(), utils.product(target_shapes.items));

        //     var iter_a = try a.dataIter();
        //     defer iter_a.deinit();

        //     var i: usize = 0;

        //     while (iter_a.next()) |idx| {
        //         const x = try a.getWithIndicesCompType(data_type, idx);
        //         const y = try c.getWithIndicesCompType(data_type, idx);

        //         new_buf[i] = op_func(x.*, y.*);
        //         i += 1;
        //     }

        //     const layout = try Layout.init(self.allocator, data_type, target_shapes.items);
        //     const storage = Storage.init(
        //         self.allocator,
        //         Storage.Device.Cpu,
        //         @ptrCast(new_buf.ptr),
        //         new_buf.len * data_type.dtypeSize(),
        //     );

        //     return try Self.fromDataImpl(self.allocator, layout, storage, 0);
        // }

        // pub fn clamp_(self: *Self, min_a: anytype, max_a: anytype) !void {
        //     const DT = comptime DataType.typeToDataType(@TypeOf(min_a));

        //     const ctx_i = .{
        //         .min = min_a,
        //         .max = max_a,
        //     };
        //     const scope = struct {
        //         fn call(v: DT.toTypeComp(), ctx: @TypeOf(ctx_i)) DT.toTypeComp() {
        //             return std.math.clamp(v, ctx.min, ctx.max);
        //         }
        //     }.call;

        //     try self.map_(DT, ctx_i, scope);
        // }

        // pub fn add_(self: *Self, value: anytype) !void {
        //     if (@TypeOf(value) == @This()) {
        //         switch (self.dtype()) {
        //             inline .bool => return error.UnsupportedType,
        //             inline else => |dt| {
        //                 const scope = struct {
        //                     fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
        //                         return v + other;
        //                     }
        //                 }.call;

        //                 return try self.binaryOp_(@as(@This(), value), dt, scope);
        //             },
        //         }
        //     }

        //     switch (self.dtype()) {
        //         inline .bool => return error.UnsupportedType,
        //         inline else => |DT| {
        //             const vv = dtype_o.toDType(DT.toTypeComp(), value);

        //             const scope = struct {
        //                 fn call(v: DT.toTypeComp(), ctx: DT.toTypeComp()) DT.toTypeComp() {
        //                     return v + ctx;
        //                 }
        //             }.call;

        //             try self.map_(DT, vv, scope);
        //         },
        //     }
        // }

        // pub fn add(self: *const Self, value: anytype) !Self {
        //     if (@TypeOf(value) == @This()) {
        //         switch (self.dtype()) {
        //             inline .bool => return error.UnsupportedType,
        //             inline else => |dt| {
        //                 const scope = struct {
        //                     fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
        //                         return v + other;
        //                     }
        //                 }.call;

        //                 return try self.binaryOp(@as(@This(), value), dt, scope);
        //             },
        //         }
        //     }

        //     switch (self.dtype()) {
        //         inline .bool => return error.UnsupportedType,
        //         inline else => |DT| {
        //             const T = DT.toTypeComp();
        //             const vv = dtype_o.toDType(T, value);

        //             const scope = struct {
        //                 fn call(v: T, ctx: T) T {
        //                     return v + ctx;
        //                 }
        //             }.call;

        //             return try self.map(DT, DT, vv, scope);
        //         },
        //     }
        // }

        // pub fn sub_(self: *Self, value: anytype) !void {
        //     if (@TypeOf(value) == @This()) {
        //         switch (self.dtype()) {
        //             inline else => |dt| {
        //                 const scope = struct {
        //                     fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
        //                         return v - other;
        //                     }
        //                 }.call;

        //                 return try self.binaryOp_(@as(@This(), value), dt, scope);
        //             },
        //         }
        //     }

        //     const DT = comptime DataType.typeToDataType(@TypeOf(value));
        //     const scope = struct {
        //         fn call(v: *DT.toTypeComp()) void {
        //             v.* -= value;
        //         }
        //     }.call;

        //     try self.map_(DT, scope);
        // }

        // pub fn sub(self: *const Self, value: anytype) !Self {
        //     if (@TypeOf(value) == @This()) {
        //         switch (self.dtype()) {
        //             inline .bool => return error.InvalidType,
        //             inline else => |dt| {
        //                 const scope = struct {
        //                     fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
        //                         return v - other;
        //                     }
        //                 }.call;

        //                 return try self.binaryOp(@as(@This(), value), dt, scope);
        //             },
        //         }
        //     }

        //     switch (self.dtype()) {
        //         inline .bool => return error.InvalidType,
        //         inline else => |DT| {
        //             const T = DT.toTypeComp();
        //             const vv = dtype_o.toDType(T, value);

        //             const scope = struct {
        //                 fn call(v: T, ctx: T) T {
        //                     return v - ctx;
        //                 }
        //             }.call;

        //             return try self.map(DT, DT, vv, scope);
        //         },
        //     }
        // }

        // pub fn mul(self: *const Self, value: anytype) !Self {
        //     if (comptime @TypeOf(value) == @This()) {
        //         switch (self.dtype()) {
        //             inline .bool => return error.InvalidType,
        //             inline else => |dt| {
        //                 const scope = struct {
        //                     fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
        //                         return v * other;
        //                     }
        //                 }.call;

        //                 return try self.binaryOp(@as(@This(), value), dt, scope);
        //             },
        //         }
        //     }

        //     switch (self.dtype()) {
        //         inline .bool => return error.InvalidType,
        //         inline else => |DT| {
        //             const T = DT.toTypeComp();
        //             const vv = dtype_o.toDType(T, value);

        //             const scope = struct {
        //                 fn call(v: T, ctx: T) T {
        //                     return v * ctx;
        //                 }
        //             }.call;

        //             return try self.map(DT, DT, vv, scope);
        //         },
        //     }
        // }

        // pub fn mul_(self: *Self, value: anytype) !void {
        //     if (@TypeOf(value) == @This()) {
        //         switch (self.dtype()) {
        //             inline else => |dt| {
        //                 const scope = struct {
        //                     fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
        //                         return v * other;
        //                     }
        //                 }.call;

        //                 return try self.binaryOp_(@as(@This(), value), dt, scope);
        //             },
        //         }
        //     }

        //     switch (self.dtype()) {
        //         inline .bool => return error.UnsupportedType,
        //         inline else => |DT| {
        //             const vv = dtype_o.toDType(DT.toTypeComp(), value);

        //             const scope = struct {
        //                 fn call(v: DT.toTypeComp(), ctx: DT.toTypeComp()) DT.toTypeComp() {
        //                     return v * ctx;
        //                 }
        //             }.call;

        //             return try self.map_(DT, vv, scope);
        //         },
        //     }
        // }

        // pub fn div_(self: *Self, value: anytype) !void {
        //     if (@TypeOf(value) == @This()) {
        //         switch (self.dtype()) {
        //             inline .f32 => |dt| {
        //                 const scope = struct {
        //                     fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
        //                         return v / other;
        //                     }
        //                 }.call;

        //                 return try self.binaryOp_(@as(@This(), value), dt, scope);
        //             },
        //             inline else => return error.UnsupportedType,
        //         }
        //     }

        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const T = DT.toTypeComp();
        //             const ctx_i = dtype_o.toDType(T, value);

        //             const func = struct {
        //                 fn call(v: T, ctx: T) T {
        //                     return v / ctx;
        //                 }
        //             }.call;

        //             try self.map_(DT, ctx_i, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn sin_(self: *Self) !void {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const func = struct {
        //                 fn call(v: DT.toTypeComp(), _: void) DT.toTypeComp() {
        //                     return @sin(v);
        //                 }
        //             }.call;
        //             try self.map_(DT, void{}, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn exp_(self: *Self) !void {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const func = struct {
        //                 fn call(v: DT.toTypeComp(), _: void) DT.toTypeComp() {
        //                     return @exp(v);
        //                 }
        //             }.call;
        //             try self.map_(DT, void{}, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn log_(self: *Self) !void {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const func = struct {
        //                 fn call(v: DT.toTypeComp()) DT.toTypeComp() {
        //                     return @log(v);
        //                 }
        //             }.call;
        //             try self.map_(DT, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn sigmoid_(self: *Self) !void {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const func = struct {
        //                 fn call(v: DT.toTypeComp()) DT.toTypeComp() {
        //                     return 1.0 / (1.0 + @exp(-v));
        //                 }
        //             }.call;
        //             return try self.map_(DT, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn relu_(self: *Self) !void {
        //     switch (self.dtype()) {
        //         inline .f32, .i32, .u32 => |DT| {
        //             const func = struct {
        //                 fn call(v: DT.toTypeComp()) DT.toTypeComp() {
        //                     return @max(v, @as(DT.toTypeComp(), 0));
        //                 }
        //             }.call;
        //             return try self.map_(DT, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn powi_(self: *Self, value: anytype) !void {
        //     switch (self.dtype()) {
        //         inline .f32, .i32, .u32 => |DT| {
        //             const ctx_i = dtype_o.toDType(DT.toTypeComp(), value);

        //             const func = struct {
        //                 fn call(v: DT.toTypeComp(), ctx: DT.toTypeComp()) DT.toTypeComp() {
        //                     return std.math.pow(DT.toTypeComp(), v, ctx);
        //                 }
        //             }.call;
        //             return try self.map_(DT, ctx_i, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn sqrt_(self: *const Self) !Self {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const func = struct {
        //                 fn call(v: DT.toTypeComp()) DT.toTypeComp() {
        //                     return @sqrt(v);
        //                 }
        //             }.call;
        //             return try self.map_(DT, func);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // // check
        // pub fn isNan(self: *const Self) !Self {
        //     switch (self.dtype()) {
        //         inline else => |DT| {
        //             const T = DT.toTypeComp();
        //             const func = struct {
        //                 fn call(v: T, _: void) bool {
        //                     return std.math.isNan(v);
        //                 }
        //             }.call;
        //             return self.mapBool(DT, void{}, func);
        //         },
        //     }
        // }

        // pub fn isInf(self: *const Self) !Self {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const T = DT.toTypeComp();
        //             const func = struct {
        //                 fn call(v: T, _: void) bool {
        //                     return std.math.isInf(v);
        //                 }
        //             }.call;
        //             return self.mapBool(DT, void{}, func);
        //         },
        //         inline else => return error.InvalidType,
        //     }
        // }

        // pub fn isPositiveInf(self: *const Self) !Self {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const T = DT.toTypeComp();
        //             const func = struct {
        //                 fn call(v: T, _: void) bool {
        //                     return std.math.isPositiveInf(v);
        //                 }
        //             }.call;
        //             return self.mapBool(DT, void{}, func);
        //         },
        //         inline else => return error.InvalidType,
        //     }
        // }

        // pub fn isNegativeInf(self: *const Self) !Self {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const T = DT.toTypeComp();
        //             const func = struct {
        //                 fn call(v: T, _: void) bool {
        //                     return std.math.isNegativeInf(v);
        //                 }
        //             }.call;
        //             return self.mapBool(DT, void{}, func);
        //         },
        //         inline else => return error.InvalidType,
        //     }
        // }

        // pub fn isFinite(self: *const Self) !Self {
        //     switch (self.dtype()) {
        //         inline .f32 => |DT| {
        //             const T = DT.toTypeComp();
        //             const func = struct {
        //                 fn call(v: T, _: void) bool {
        //                     return std.math.isFinite(v);
        //                 }
        //             }.call;
        //             return self.mapBool(DT, void{}, func);
        //         },
        //         inline else => return error.InvalidType,
        //     }
        // }

        // // may meet bug when use inplace
        // pub fn nanToNum_(self: *Self, nan: anytype, args: struct { posinf: ?@TypeOf(nan) = null, neginf: ?@TypeOf(nan) = null }) !void {
        //     switch (self.dtype()) {
        //         inline else => return error.InvalidType,
        //         inline .f32 => |DT| {
        //             const T = DT.toTypeComp();

        //             const Ctx = struct {
        //                 nan: T,
        //                 posinf: ?T,
        //                 neginf: ?T,
        //             };

        //             const ctx_i = Ctx{
        //                 .nan = dtype_o.toDType(T, nan),
        //                 .posinf = if (args.posinf) |posinf| dtype_o.toDType(T, posinf) else null,
        //                 .neginf = if (args.neginf) |neginf| dtype_o.toDType(T, neginf) else null,
        //             };

        //             const func = struct {
        //                 fn call(v: T, ctx: Ctx) T {
        //                     if (std.math.isNan(v)) {
        //                         return ctx.nan;
        //                     } else if (std.math.isPositiveInf(v)) {
        //                         return if (ctx.posinf) |posinf| posinf else v;
        //                     } else if (std.math.isNegativeInf(v)) {
        //                         return if (ctx.neginf) |neginf| neginf else v;
        //                     } else {
        //                         return v;
        //                     }
        //                 }
        //             }.call;
        //             return try self.map_(DT, ctx_i, func);
        //         },
        //     }
        // }

        // pub fn softmax(self: *const Self) !Self {
        //     const dims = self.ndim();

        //     if (dims == 0) {
        //         return error.InvalidDimension;
        //     }
        //     const a = try self.max(dims - 1);
        //     var v = try self.sub(try a.unsqueeze(dims - 1));
        //     try v.exp_();

        //     const v1 = try (try v.sum(dims - 1)).unsqueeze(dims - 1);

        //     std.debug.print("v: {f} v1: {f}\n", .{ v, v1 });
        //     try v.div_(v1);

        //     return v;
        // }

        // //
        // //
        // //
        // // reduce method
        // pub fn reduce(self: *const Self, comptime data_type: DataType, dim: ?usize, op_func: fn (acc: data_type.toTypeComp(), x: data_type.toTypeComp()) data_type.toTypeComp(), post_func: ?fn (acc: data_type.toTypeComp(), count: usize) data_type.toTypeComp()) !Self {
        //     const T = data_type.toTypeComp();

        //     const indices_init = try self.allocator.alloc(usize, self.ndim());
        //     defer self.allocator.free(indices_init);
        //     for (indices_init) |*i| i.* = 0;

        //     const op_init = (try self.getWithIndicesCompType(data_type, indices_init)).*;

        //     if (dim) |dm| {
        //         var shapes_i = try std.ArrayList(usize).initCapacity(self.allocator, self.ndim() - 1);
        //         try shapes_i.appendSlice(self.allocator, self.shapes());
        //         _ = shapes_i.orderedRemove(dm);

        //         const data_len = utils.product(shapes_i.items);
        //         var new_buf = try self.allocator.alloc(T, data_len);

        //         var indices = try self.allocator.alloc(usize, self.ndim());
        //         defer self.allocator.free(indices);

        //         for (indices) |*i| i.* = 0;

        //         var out_i: usize = 0;
        //         var done = false;

        //         while (!done) {
        //             var acc: T = op_init;
        //             for (0..self.shapes()[dm]) |k| {
        //                 indices[dm] = k;

        //                 if (!std.mem.eql(usize, indices, indices_init)) {
        //                     acc = op_func(acc, (try self.getWithIndicesCompType(data_type, indices)).*);
        //                 }
        //             }

        //             if (post_func) |pf| {
        //                 acc = pf(acc, self.shapes()[dm]);
        //             }

        //             new_buf[out_i] = acc;
        //             out_i += 1;

        //             var j = self.shapes().len;

        //             if (j == 0) {
        //                 break;
        //             }

        //             while (j > 0) : (j -= 1) {
        //                 if (j - 1 == dm) {
        //                     if (dm == 0) {
        //                         done = true;
        //                     }
        //                     continue;
        //                 }

        //                 if (indices[j - 1] < self.shapes()[j - 1] - 1) {
        //                     indices[j - 1] += 1;
        //                     break;
        //                 } else {
        //                     indices[j - 1] = 0;

        //                     if (j - 1 == 0) {
        //                         done = true;
        //                     }
        //                 }
        //             }
        //         }

        //         const layout = try Layout.init(self.allocator, data_type, shapes_i.items);

        //         const bytes_size = new_buf.len * @sizeOf(T);
        //         const storage = Storage.init(self.allocator, Storage.Device.Cpu, @ptrCast(new_buf.ptr), bytes_size);

        //         return try Self.fromDataImpl(self.allocator, layout, storage, 0);
        //     } else {
        //         var total: T = op_init;

        //         var idx = try self.allocator.alloc(usize, self.shapes().len);
        //         for (idx) |*x| x.* = 0;

        //         var done = false;

        //         var count: usize = 0;
        //         while (!done) {
        //             if (!std.mem.eql(usize, idx, indices_init)) {
        //                 total = op_func(total, (try self.getWithIndicesCompType(data_type, idx)).*);
        //             }

        //             count += 1;

        //             var d: usize = self.shapes().len;
        //             while (d > 0) : (d -= 1) {
        //                 idx[d - 1] += 1;

        //                 if (idx[d - 1] < self.shapes()[d - 1]) {
        //                     break;
        //                 } else if (d == 1) {
        //                     done = true;
        //                 } else {
        //                     idx[d - 1] = 0;
        //                 }
        //             }
        //         }

        //         if (post_func) |pf| {
        //             total = pf(total, count);
        //         }

        //         var new_buf = try self.allocator.alloc(T, 1);
        //         new_buf[0] = total;

        //         const layout = try Layout.init(self.allocator, data_type, &.{});
        //         const storage = Storage.init(self.allocator, Storage.Device.Cpu, @ptrCast(new_buf.ptr), @sizeOf(T));

        //         return try Self.fromDataImpl(self.allocator, layout, storage, 0);
        //     }
        // }

        // pub fn sum(self: *const Self, dim: ?usize) !Self {
        //     switch (self.dtype()) {
        //         inline .f32, .i32, .u32 => |dt| {
        //             const T = dt.toTypeComp();
        //             const scope = struct {
        //                 fn op_func(acc: T, val: T) T {
        //                     return acc + val;
        //                 }
        //             };
        //             return try self.reduce(dt, dim, scope.op_func, null);
        //         },
        //         inline else => return error.UnsupportedType,
        //     }
        // }

        // pub fn max(self: *const Self, dim: ?usize) !Self {
        //     switch (self.dtype()) {
        //         inline else => |v| {
        //             const T = v.toTypeComp();
        //             const scope = struct {
        //                 fn op_func(acc: T, val: T) T {
        //                     return @max(acc, val);
        //                 }
        //             };
        //             return try self.reduce(v, dim, scope.op_func, null);
        //         },
        //         inline .bool => return error.UnsupportedType,
        //     }
        // }

        // pub fn min(self: *const Self, dim: ?usize) !Self {
        //     switch (self.dtype()) {
        //         inline .bool => return error.UnsupportedType,
        //         inline else => |dt| {
        //             const T = dt.toTypeComp();
        //             const scope = struct {
        //                 fn op_func(acc: T, val: T) T {
        //                     return @min(acc, val);
        //                 }
        //             };
        //             return try self.reduce(dt, dim, std.math.inf(T), scope.op_func, null);
        //         },
        //     }
        // }

        // pub fn prod(self: *const Self, dim: ?usize) !Self {
        //     switch (self.dtype()) {
        //         inline .bool => return error.UnsupportedType,
        //         inline else => |dt| {
        //             const T = dt.toTypeComp();
        //             const scope = struct {
        //                 fn op_func(acc: T, val: T) T {
        //                     return acc * val;
        //                 }
        //             };
        //             return try self.reduce(dt, dim, std.math.inf(T), scope.op_func, null);
        //         },
        //     }
        // }

        // pub fn mean(self: *const Self, dim: ?usize) !Self {
        //     switch (self.dtype()) {
        //         inline else => return error.UnsupportedType,
        //         inline .f16, .f32 => |dt| {
        //             const T = dt.toTypeComp();
        //             const scope = struct {
        //                 fn op_func(acc: T, val: T) T {
        //                     return acc + val;
        //                 }
        //                 fn post_func(acc: T, count: usize) T {
        //                     return acc / @as(T, @floatFromInt(count));
        //                 }
        //             };

        //             return try self.reduce(dt, dim, scope.op_func, scope.post_func);
        //         },
        //     }
        // }

        // pub fn any(self: *const Self, dim: ?usize) !Self {
        //     switch (self.dtype()) {
        //         inline else => return error.UnsupportedType,
        //         inline .bool => |dt| {
        //             const scope = struct {
        //                 fn op_func(acc: bool, val: bool) bool {
        //                     return acc or val;
        //                 }
        //             };

        //             return try self.reduce(dt, dim, scope.op_func, null);
        //         },
        //     }
        // }

        // pub fn all(self: *const Self, dim: ?usize) !Self {
        //     switch (self.dtype()) {
        //         inline else => return error.UnsupportedType,
        //         inline .bool => |dt| {
        //             const scope = struct {
        //                 fn op_func(acc: bool, val: bool) bool {
        //                     return acc and val;
        //                 }
        //             };

        //             return try self.reduce(dt, dim, scope.op_func, null);
        //         },
        //     }
        // }

        // // op method
        // pub fn matmul(self: *const Self, other: *const Self) anyerror!Self {
        //     if (self.dtype() != other.dtype()) {
        //         return error.TypeMismatch;
        //     }

        //     if (self.ndim() != 2 or other.ndim() != 2) {
        //         return error.ShapeMismatch;
        //     }

        //     if (self.dtype() != DataType.f32) {
        //         return error.TypeNotSupported;
        //     }

        //     if (self.shapes()[1] != other.shapes()[0]) {
        //         return error.ShapeMismatch;
        //     }

        //     const lhs = if (!self.layout.isContiguous()) &(try self.contiguous()) else self;

        //     const rhs = if (!other.layout.isContiguous()) &(try other.contiguous()) else other;

        //     const m = lhs.shapes()[0];
        //     const n = rhs.shapes()[1];
        //     const k = lhs.shapes()[1];

        //     const a: [*c]const f32 = @ptrCast(lhs.storage.dataSlice(f32));
        //     const b: [*c]const f32 = @ptrCast(rhs.storage.dataSlice(f32));

        //     const buf = try std.ArrayList(f32).initCapacity(lhs.allocator, m * n);
        //     const c = @as([*]f32, @ptrCast(buf.items.ptr));

        //     host.matmul(a, b, c, m, n, k);

        //     const data = @as([*]u8, @ptrCast(c));

        //     return try Self.fromDataRaw(lhs.allocator, DataType.f32, &.{ m, n }, Storage.Device.Cpu, data, m * n * @sizeOf(f32));
        // }

        // // iterate method
        // pub fn dataIter(self: *const Self) !ShapeIterator {
        //     return try ShapeIterator.init(self.allocator, self.shapes());
        // }

        // create method
        pub fn fromDataImpl(layout_a: Layout, storage_a: Storage, storage_offset_a: usize) !Self {
            return Self{
                .layout = layout_a,
                .storage = storage_a,
                ._storage_offset = storage_offset_a,
            };
        }

        // pub fn fromDataRaw(allocator: std.mem.Allocator, dtype_i: DataType, shapes_a: []const usize, device: Storage.Device, data: [*]u8, bytes_size: usize) anyerror!Self {
        //     const storage = Storage.init(allocator, device, data, bytes_size);

        //     const layout = try Layout.init(allocator, dtype_i, shapes_a);

        //     return try Self.fromDataImpl(allocator, layout, storage, 0);
        // }

        // pub fn fromData(allocator: std.mem.Allocator, comptime dtype_i: DataType, shapes_a: []const usize, data: std.ArrayList(dtype_i.toTypeComp())) anyerror!Self {
        //     const buf_r: [*]u8 = @ptrCast(data.items.ptr);
        //     const bytes_size = dtype_i.dtypeSize() * data.items.len;

        //     return Self.fromDataRaw(allocator, dtype_i, shapes_a, Storage.Device.Cpu, buf_r, bytes_size);
        // }

        // pub fn fromSlice(allocator: std.mem.Allocator, comptime dtype_i: DataType, shapes_a: []const usize, data: []const dtype_i.toTypeComp()) anyerror!Self {
        //     var data_list = try std.ArrayList(dtype_i.toTypeComp()).initCapacity(allocator, data.len);
        //     try data_list.appendSlice(allocator, data);

        //     return try Self.fromData(allocator, dtype_i, shapes_a, data_list);
        // }

        // pub fn to(self: *const Self, data_type: DataType) !Self {
        //     if (data_type == self.dtype()) {
        //         const layout = try self.layout.clone();
        //         const storage = self.storage.clone();

        //         return try Self.fromDataImpl(self.allocator, layout, storage, self._storage_offset);
        //     } else {
        //         switch (data_type) {
        //             inline else => |dt| {
        //                 const layout = try Layout.init(self.allocator, dt, self.shapes());

        //                 var new_buf = try self.allocator.alloc(dt.toTypeComp(), self.layout.size());

        //                 var iter = try self.dataIter();

        //                 switch (self.dtype()) {
        //                     inline else => |sdt| {
        //                         var i: usize = 0;
        //                         while (iter.next()) |idx| {
        //                             switch (dt) {
        //                                 .f16, .f32 => switch (sdt) {
        //                                     .f16, .f32 => {
        //                                         new_buf[i] = @floatCast((try self.getWithIndicesCompType(sdt, idx)).*);
        //                                     },
        //                                     .i32, .u32 => {
        //                                         new_buf[i] = @floatFromInt((try self.getWithIndicesCompType(sdt, idx)).*);
        //                                     },
        //                                     .bool => {
        //                                         new_buf[i] = if ((try self.getWithIndicesCompType(sdt, idx)).*) 1.0 else 0.0;
        //                                     },
        //                                 },
        //                                 .i32, .u32 => switch (sdt) {
        //                                     .f16, .f32 => {
        //                                         new_buf[i] = @intFromFloat((try self.getWithIndicesCompType(sdt, idx)).*);
        //                                     },
        //                                     .i32, .u32 => {
        //                                         new_buf[i] = @intCast((try self.getWithIndicesCompType(sdt, idx)).*);
        //                                     },
        //                                     .bool => {
        //                                         new_buf[i] = if ((try self.getWithIndicesCompType(sdt, idx)).*) 1 else 0;
        //                                     },
        //                                 },
        //                                 .bool => switch (sdt) {
        //                                     .f16, .f32 => {
        //                                         new_buf[i] = (try self.getWithIndicesCompType(sdt, idx)).* > 0.0;
        //                                     },
        //                                     .i32, .u32 => {
        //                                         new_buf[i] = (try self.getWithIndicesCompType(sdt, idx)).* > 0;
        //                                     },
        //                                     .bool => {
        //                                         new_buf[i] = (try self.getWithIndicesCompType(sdt, idx)).*;
        //                                     },
        //                                 },
        //                             }

        //                             i += 1;
        //                         }

        //                         const storage = Storage.init(self.allocator, Storage.Device.Cpu, @ptrCast(new_buf.ptr), new_buf.len * data_type.dtypeSize());

        //                         return Self.fromDataImpl(self.allocator, layout, storage, 0);
        //                     },
        //                 }
        //             },
        //         }
        //     }
        // }

        // pub fn clone(self: *const Self) !Self {
        //     const layout = try self.layout.clone();
        //     const storage = try self.storage.deepCopy();

        //     return try Self.fromDataImpl(self.allocator, layout, storage, 0);
        // }

        // pub fn fromShapedData(allocator: std.mem.Allocator, comptime arr: anytype) anyerror!Self {
        //     const T = utils.getArrayRefItemType(@TypeOf(arr));
        //     const dtype_i = comptime DataType.typeToDataType(T);

        //     const shapes_i = utils.getArrayRefShapes(@TypeOf(arr));
        //     const data_len = utils.product(shapes_i);
        //     const new_buf = try allocator.alloc(T, data_len);

        //     const arr1: [*]const T = @ptrCast(arr);
        //     @memcpy(new_buf, arr1);

        //     return Self.fromDataRaw(allocator, dtype_i, shapes_i, Storage.Device.Cpu, @ptrCast(new_buf.ptr), new_buf.len * @sizeOf(T));
        // }

        // pub fn contiguous(self: Self) !Self {
        //     if (self.layout.isContiguous()) {
        //         return self;
        //     }

        //     const elem_size = self.dtype().dtypeSize();

        //     const new_buf = try self.allocator.alloc(u8, self.byteSize() * elem_size);

        //     var idx: usize = 0;
        //     const indices = try self.allocator.alloc(usize, self.ndim());

        //     const inner_scope = struct {
        //         fn copyRecursive(tensor: *const Self, indices_i: []usize, dim: usize, new_buf_i: []u8, idx_i: *usize, elem_size_a: usize) void {
        //             if (dim == tensor.ndim()) {
        //                 var offset: usize = tensor._storage_offset;
        //                 for (indices_i, 0..) |ind, i| {
        //                     offset += ind * tensor.strides()[i];
        //                 }
        //                 offset *= elem_size_a;

        //                 const src = tensor.rawDataSlice()[offset .. offset + elem_size_a];
        //                 const dst = new_buf_i[idx_i.* * elem_size_a .. (idx_i.* + 1) * elem_size_a];
        //                 @memcpy(dst, src);

        //                 idx_i.* += 1;
        //                 return;
        //             }

        //             const shape_dim = tensor.shapes()[dim];
        //             for (0..shape_dim) |i| {
        //                 indices_i[dim] = i;
        //                 copyRecursive(tensor, indices_i, dim + 1, new_buf_i, idx_i, elem_size_a);
        //             }
        //         }
        //     };

        //     inner_scope.copyRecursive(&self, indices, 0, new_buf, &idx, elem_size);

        //     var shapes_a = try std.ArrayList(usize).initCapacity(self.allocator, self.ndim());
        //     try shapes_a.appendSlice(self.allocator, self.shapes());

        //     return Self.fromDataRaw(self.allocator, self.layout.dtype(), shapes_a.items, Storage.Device.Cpu, @as([*]u8, @ptrCast(new_buf.ptr)), self.storage.byteSize());
        // }

        // // attributes

        // pub fn broadcastTo(self: *const Self, target_shape: []const usize) !Self {
        //     var t = try self.clone();
        //     try t.broadcastTo_(target_shape);
        //     return t;
        // }

        // pub fn broadcastTo_(self: *Self, target_shape: []const usize) !void {
        //     const new_strides = try utils.broadcastShapes(self.allocator, self.shapes(), self.strides(), target_shape);

        //     var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, target_shape.len);
        //     try new_shapes.appendSlice(self.allocator, target_shape);

        //     const layout = try Layout.initRaw(self.allocator, self.dtype(), new_shapes, new_strides);

        //     self.layout = layout;
        // }

        pub fn get(self: *const Self, indices: [N]usize) !T {
            var idx = try utils.indices_to_flat(&indices, &self.shape(), &self.stride());
            idx += self._storage_offset;

            return self.storage.dataSlice()[idx];
        }

        pub fn set(self: *Self, indices: [N]usize, value: T) !void {
            var idx = try utils.indices_to_flat(indices, self.shape(), self.stride());
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
        pub fn transpose_(self: *Self) !void {
            self.layout = try self.layout.transpose(0, 1);
        }

        pub fn permute_(self: *const Self, perm: [N]usize) !void {
            self.layout = try self.layout.permute(perm);
        }

        pub fn reshape_(self: *const Self, new_shapes: []const usize) !void {
            self.layout = try self.layout.reshape(new_shapes);
        }

        pub fn unsqueeze_(self: *const Self, dim: usize) !void {
            self.layout = try self.layout.unsqueeze(dim);
        }

        pub fn squeeze_(self: *const Self, dim: usize) !void {
            self.layout = try self.layout.squeeze(dim);
        }

        // create factory method
        pub fn arange(allocator: std.mem.Allocator, args: struct {
            start: T = @as(T, 0),
            step: T = @as(T, 1),
            end: T,
        }) !Self {
            if (N != 1) @compileError("arange is only supported for 1D tensors");
            const storage = try Storage.arange(allocator, .{
                .start = args.start,
                .step = args.step,
                .end = args.end,
            });
            const layout = Layout.init([1]usize{storage.len()});

            return Self.fromDataImpl(layout, storage, 0);
        }

        pub fn linspace(allocator: std.mem.Allocator, args: struct {
            start: T,
            end: T,
            steps: usize,
        }) !Self {
            if (N != 1) @compileError("arange is only supported for 1D tensors");
            const storage = try Storage.linspace(allocator, .{
                .start = args.start,
                .end = args.end,
                .steps = args.steps,
            });
            const layout = Layout.init([1]usize{storage.len()});

            return Self.fromDataImpl(layout, storage, 0);
        }

        // core method
        pub fn deinit(self: *const Self) void {
            self.storage.deinit();
        }

        pub fn size(self: *const Self) usize {
            return self.layout.size();
        }

        pub fn ndim(self: *const Self) usize {
            return self.layout.ndim();
        }

        pub fn shape(self: *const Self) [N]usize {
            return self.layout.shape();
        }

        pub fn stride(self: *const Self) [N]usize {
            return self.layout.stride();
        }

        pub fn equal(self: *const Self, other: *const Self) bool {
            if (!self.layout.equal(&other.layout)) return false;
            if (self.dtype() != other.dtype()) return false;

            var self_iter = self.dataIter() catch unreachable;

            while (self_iter.next()) |idx| {
                const sv = self.getWithIndices(idx) catch unreachable;
                const ov = other.getWithIndices(idx) catch unreachable;

                if (!sv.equal(ov)) return false;
            }

            return true;
        }

        pub fn approxEqual(self: *const Self, other: *const Self, comptime dtype_i: DataType, relEps: dtype_i.toType(), absEps: dtype_i.toType()) bool {
            if (!self.layout.equal(&other.layout)) return false;

            if (self.dtype() != other.dtype()) return false;

            if (self.dtype() != dtype_i) return false;

            const self_data_slice = self.dataSlice(dtype_i.toType());
            const other_data_slice = other.dataSlice(dtype_i.toType());
            return utils.sliceApproxEqual(dtype_i.toType(), self_data_slice, other_data_slice, relEps, absEps);
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print(
                \\Tensor{{
                \\.DType = {}
                \\.{f}
                \\.Data =
            , .{ T, self.layout });

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
                try writer.print("{}", .{try self.get(indices)});
            } else if (depth == dims - 1) {
                try self.fmt1dSlice(writer, depth, indices);
            } else {
                try self.fmtNdSlice(writer, depth, indices);
            }
        }

        fn fmtNdSlice(self: *const Self, writer: *std.Io.Writer, depth: usize, base_indices: [N]usize) anyerror!void {
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

                    try writer.print("{}", .{try self.get(idx)});
                }
            } else {
                for (0..pad_show_count) |i| {
                    if (i > 0) {
                        _ = try writer.write(" ");
                    }

                    var idx = base_indices;
                    idx[depth] = i;

                    try writer.print("{}", .{try self.get(idx)});
                }
                _ = try writer.write(" ... ");

                for (current_dim_size - pad_show_count..current_dim_size) |i| {
                    var idx = base_indices;
                    idx[depth] = i;

                    try writer.print("{}", .{try self.get(idx)});

                    if (i < current_dim_size - 1) {
                        _ = try writer.write(" ");
                    }
                }
            }

            _ = try writer.write("]");
        }
    };
}

pub fn full(allocator: std.mem.Allocator, shapes_a: anytype, value: anytype) !Tensor(
    utils.getCompArrayLen(@TypeOf(shapes_a)),
    .{ .T = @TypeOf(value) },
) {
    const T = @TypeOf(value);
    const N = comptime shapes_a.len;

    const Layout = layout_t.Layout(N);
    const Storage = storage_t.Storage(T, .Cpu);
    const TensorI = Tensor(N, .{ .T = T });

    const element_count = utils.product(&shapes_a);

    const layout = Layout.init(shapes_a);
    const storage = try Storage.full(allocator, element_count, value);

    return TensorI.fromDataImpl(layout, storage, 0);
}

pub fn fullLike(allocator: std.mem.Allocator, tensor: anytype, value: @TypeOf(tensor).T) !@TypeOf(tensor) {
    return try full(allocator, tensor.shape(), value);
}

pub fn zeros(allocator: std.mem.Allocator, shapes_a: anytype) !Tensor(
    utils.getCompArrayLen(@TypeOf(shapes_a)),
    .{},
) {
    const value: f32 = 0;

    return try full(allocator, shapes_a, value);
}

pub fn zerosLike(allocator: std.mem.Allocator, tensor: anytype) !@TypeOf(tensor) {
    return try zeros(allocator, tensor.shapes());
}

pub fn ones(allocator: std.mem.Allocator, shapes_a: anytype) !Tensor(
    utils.getCompArrayLen(@TypeOf(shapes_a)),
    .{},
) {
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
        tensor.set([2]usize{ i, i }, value);
    }

    return tensor;
}

pub fn rand(allocator: std.mem.Allocator, shapes_a: anytype, low: anytype, high: @TypeOf(low)) !Tensor(
    utils.getCompArrayLen(@TypeOf(shapes_a)),
    .{ .T = utils.floatBasicType(@TypeOf(low)) },
) {
    const N = comptime utils.getCompArrayLen(@TypeOf(shapes_a));
    const T = comptime switch (@typeInfo(@TypeOf(low))) {
        inline .float => |DT| DT,
        inline .comptime_float => f64,
        inline else => @compileError("only support f32 and f64"),
    };

    const layout = layout_t.Layout(N).init(shapes_a);
    const size = layout.size();

    const storage = try storage_t.Storage(T, .Cpu).rand(allocator, size, low, high);
    return try Tensor(N, .{ .T = T }).fromDataImpl(layout, storage, 0);
}

pub fn randNorm(allocator: std.mem.Allocator, shapes_a: anytype, mean_a: anytype, stddev: @TypeOf(mean_a)) !Tensor(
    utils.getCompArrayLen(@TypeOf(shapes_a)),
    .{ .T = utils.floatBasicType(@TypeOf(mean_a)) },
) {
    const N = comptime utils.getCompArrayLen(@TypeOf(shapes_a));
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

test "tensor create" {
    const allocator = std.testing.allocator;

    const Tensor1 = Tensor(1, .{ .T = f32 });

    {
        const t1 = try Tensor1.arange(allocator, .{ .start = 0, .step = 2, .end = 10 });
        defer t1.deinit();
        std.debug.print("t1: {f}\n", .{t1});
    }

    {
        const t2 = try Tensor1.linspace(allocator, .{ .start = 7, .end = 30, .steps = 5 });
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
        for (result) |t| {
            defer t.deinit();
            std.debug.print("unbind t: {f}\n", .{t});
        }
    }
}

// test "shape test" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const arr1 = [3][2]f32{
//         [2]f32{ 1.0, 2.0 },
//         [2]f32{ 3.0, 4.0 },
//         [2]f32{ 5.0, 6.0 },
//     };
//     const t111 = try Self.fromShapedData(allocator, &arr1);

//     const t111_transposed = try t111.transpose_();

//     try std.testing.expect(t111.shapes()[0] == t111_transposed.shapes()[1]);
//     try std.testing.expect(t111.shapes()[1] == t111_transposed.shapes()[0]);
//     try std.testing.expectEqual(t111.getWithIndices(&.{ 0, 1 }), t111_transposed.getWithIndices(&.{ 1, 0 }));

//     std.debug.print("t111: {f} t111 transposed: {f}\n", .{ t111, t111_transposed });

//     // const arr2 = [2][4]f32{
//     //
//     //     [4]f32{ 3.0, 4.0, 5.0, 6.0 },
//     //     [4]f32{ 5.0, 6.0, 7.0, 8.0 },
//     // };
//     // const t112 = try Tensor.fromShapedData(allocator, &arr2);
//     const t111_unsqueezed = try t111.unsqueeze(1);
//     try std.testing.expectEqualSlices(usize, t111_unsqueezed.shapes(), &.{ 3, 1, 2 });
//     const t111_squeezed = try t111_unsqueezed.squeeze(null);
//     try std.testing.expectEqualSlices(usize, t111_squeezed.shapes(), &.{ 3, 2 });

//     std.debug.print("unsqueezed: {f} squeezed: {f}\n", .{ t111_unsqueezed, t111_squeezed });
// }

// test "random test" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.rand(allocator, f32, &.{ 3000, 3000 }, 0.0, 1.0);
//     std.debug.print("t1: {f}\n", .{t1});

//     var t2 = try Self.randNorm(allocator, f32, &.{ 3000, 3000 }, 0.0, 1.0);
//     std.debug.print("t2: {f}\n", .{t2});

//     const t2_tc = try (try t2.transpose()).contiguous();

//     const begin = std.time.milliTimestamp();
//     const t3 = try t1.matmul(&t2_tc);
//     const end = std.time.milliTimestamp();

//     std.debug.print("t3: {f}\nelapsed: {d} microseconds\n", .{ t3, end - begin });
// }

// test "permute test" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.rand(allocator, &.{ 1, 2, 3, 4, 5, 6 }, 0.0, 2.0);
//     const t1p = try t1.permute(&.{ 5, 4, 3, 2, 1, 0 });

//     std.debug.print("t1: {f}\nt1p: {f}\n", .{ t1.layout, t1p.layout });
// }

// test "contiguous test" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.fromSlice(allocator, DataType.f32, &.{ 3, 4 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
//     std.debug.print("t1: {f}\n", .{t1});

//     const t1_ds = t1.dataSlice(f32);

//     std.debug.print("t1 ds: {any}\n", .{t1_ds});

//     const t1t = try t1.transpose_();
//     std.debug.print("t1t: {f}\n", .{t1t});

//     const t1tc = try t1t.contiguous();
//     std.debug.print("t1tc: {f}\n", .{t1tc});
//     try std.testing.expect(t1tc.layout.isContiguous());

//     std.debug.print("t1t ds: {any}\nt1tc ds: {any}\n", .{ t1t.dataSlice(f32), t1tc.dataSlice(f32) });

//     try std.testing.expectApproxEqAbs((try t1t.getWithIndicesCompType(DataType.f32, &.{ 0, 2 })).*, (try t1tc.getWithIndicesCompType(DataType.f32, &.{ 0, 2 })).*, 0.00001);

//     // var shape_1 = try std.ArrayList(usize).initCapacity(allocator, 10);
//     // try shape_1.appendSlice(allocator, &.{3, 4});

//     // const data = try std.ArrayList(f32).initCapacity(allocator, num: usize)
//     // const t1 = try Tensor.fromData(allocator: Allocator, comptime dtype_i: DataType, shapes_a: Aligned(usize), data: Aligned(either type))
// }

// test "cat stack" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.rand(allocator, &.{ 1, 2, 2 }, 0.0, 2.0);

//     const t2 = try Self.rand(allocator, &.{ 1, 2, 2 }, 0.0, 2.0);

//     const t3 = try Self.cat(allocator, &.{ t1, t2 }, 2);

//     const t4 = try Self.stack(allocator, &.{ t1, t2 }, 2);
//     std.debug.print("t1: {f} t2: {f} t3: {f} t4: {f}\n", .{ t1.layout, t2.layout, t3.layout, t4.layout });
// }

// test "split unbind" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t = try Self.arange(allocator, DataType.f32, .{ .end = 20 });
//     std.debug.print("t: {f}\n", .{t});

//     const t1 = try t.reshape(&.{ 2, 2, 5 });

//     {
//         const results = try t1.split(2, 2);

//         std.debug.print("t1: {f}\n", .{t1});
//         for (results) |result| {
//             std.debug.print("result: {f} offset= {}\n", .{ result, result._storage_offset });
//         }
//     }

//     std.debug.print("begin chunk\n", .{});
//     {
//         const results = try t1.chunk(5, 2);

//         std.debug.print("t1: {f}\n", .{t1});
//         for (results) |result| {
//             std.debug.print("result: {f} offset= {}\n", .{ result, result._storage_offset });
//         }
//     }

//     std.debug.print("begin unbind\n", .{});
//     {
//         const results = try t1.unbind(2);

//         std.debug.print("t1: {f}\n", .{t1});
//         for (results) |result| {
//             std.debug.print("result: {f} offset= {} contiguoused= {f}\n", .{ result, result._storage_offset, try result.contiguous() });
//         }
//     }
// }

// test "map" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     var t = try Self.arange(allocator, DataType.f32, .{ .end = 10 });

//     const func = struct {
//         fn call(x: f32, _: void) f32 {
//             return x * 3;
//         }
//     }.call;
//     try t.map_(DataType.f32, void{}, func);
//     std.debug.print("t: {f}\n", .{t});

//     const a: f32 = 11.0;

//     try t.add_(a);
//     try t.add_(t);
//     std.debug.print("add t: {f}\n", .{t});

//     try t.mul_(2.0);
//     std.debug.print("mul t: {f}\n", .{t});

//     try t.sin_();
//     try t.exp_();

//     std.debug.print("t: {f}\n", .{t});

//     try t.clamp_(0.0, 2.39);
//     std.debug.print("t: {f}\n", .{t});

//     const t1 = try t.add(t);
//     std.debug.print("t: {f} t1: {f}\n", .{ t, t1 });

//     const a1 = try Self.rand(allocator, &.{ 1, 3 }, 0.0, 2.0);
//     const a2 = try Self.rand(allocator, &.{ 3, 1 }, -2.0, 5.0);

//     const a3 = try a1.add(a2);
//     std.debug.print("a1: {f} a2: {f} a3: {f}\n", .{ a1, a2, a3 });

//     const a4 = try a3.add(10);
//     std.debug.print("a4: {f}\n", .{a4});

//     const a5 = try a4.sub(10);
//     std.debug.print("a5: {f}\n", .{a5});

//     const a6 = try a5.mul(10);
//     std.debug.print("a6: {f}\n", .{a6});
// }

// test "reduce" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     var a1 = try std.ArrayList(usize).initCapacity(allocator, 10);
//     try a1.appendNTimes(allocator, 10, 10);
//     std.debug.print("a1: {any}\n", .{a1});
//     _ = a1.orderedRemove(0);

//     std.debug.print("a1: {any}\n", .{a1});

//     {
//         const t1 = try Self.arange(allocator, DataType.f32, .{ .end = 10 });
//         const t2 = try t1.sum(0);
//         const t3 = try t1.mean(0);
//         std.debug.print("t1: {f} t2: {f} t2 item: {} t3: {f}\n", .{ t1, t2, try t2.scalarItemComp(DataType.f32), t3 });
//     }

//     {
//         const t1 = try (try Self.arange(allocator, DataType.f32, .{ .end = 10 })).reshape(&.{ 2, 5 });
//         const t2 = try t1.sum(1);
//         const t3 = try t1.mean(1);
//         std.debug.print("t1: {f} t2: {f} t3: {f}\n", .{ t1, t2, t3 });
//     }
// }

// test "binary op" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t = try Self.arange(allocator, DataType.f32, .{ .end = 10 });
//     std.debug.print("typ: {any}\n", .{@typeInfo(@TypeOf(&t))});
// }

// test "iterator" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t = try Self.arange(allocator, DataType.f32, .{ .end = 10 });
//     const t1 = try t.reshape(&.{ 2, 5 });

//     var iter1 = try t1.dataIter();
//     while (iter1.next()) |item| {
//         std.debug.print("item: {any}\n", .{item});
//     }
//     // std.debug.print("typ: {any}\n", .{@typeInfo(@TypeOf(&t))});

// }

// test "bool op" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.rand(allocator, &.{ 2, 3 }, -1.0, 1.0);
//     std.debug.print("t1: {f}\n", .{t1});

//     const t2 = try t1.eql(0.0);
//     const t3 = try t1.lt(0.0);
//     const t4 = try t1.gt(0.0);
//     std.debug.print("t1: {f} t2: {f} t3: {f} t4: {f}\n", .{ t1, t2, t3, t4 });
// }

// test "to" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.rand(allocator, &.{ 2, 3 }, -1.0, 1.0);
//     std.debug.print("t1: {f}\n", .{t1});

//     // const a1: f32 = 20.01;
//     const t2 = try t1.to(DataType.f16);

//     std.debug.print("t2: {f}\n", .{t2});

//     const t3 = try t2.to(DataType.bool);

//     std.debug.print("t3: {f}\n", .{t3});
// }

// test "create functions" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.full(allocator, &.{ 3, 5 }, 10);
//     const t2 = try Self.linspace(allocator, DataType.f32, .{ .start = -27, .end = 33, .steps = 10 });
//     const t3 = try Self.eye(allocator, DataType.f32, 4, 5);

//     std.debug.print("t1: {f} t2: {f} t3: {f}\n", .{ t1, t2, t3 });
// }

// test "activation function" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const t1 = try Self.rand(allocator, &.{ 2, 5 }, -1.0, 1.0);

//     var t2 = try t1.clone();
//     try t2.sigmoid_();
//     var t3 = try t1.clone();
//     try t3.relu_();
//     const t4 = try t1.max(1);
//     const t5 = try t1.max(0);
//     std.debug.print("t1: {f} t2: {f} t3: {f} t4: {f} t5: {f}\n", .{ t1, t2, t3, t4, t5 });

//     const arr = [3]f32{ 0.3, 2.9, 4.0 };
//     const v = try Self.fromShapedData(allocator, &arr);
//     var v1 = try v.softmax();
//     const v2 = try v1.sum(null);

//     std.debug.print("v: {f} v1: {f} v2: {f}\n", .{ v, v1, v2 });
// }

// test "masked" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     var t1 = try Self.rand(allocator, &.{ 3, 5 }, -1.0, 1.0);

//     const arr1 = [3][5]bool{ .{ true, false, true, false, true }, .{ false, false, false, true, false }, .{ false, true, false, true, false } };
//     const m1 = try Self.fromShapedData(allocator, &arr1);

//     std.debug.print("t1: {f}\n", .{t1});

//     try t1.maskedFill_(m1, 0.0);
//     std.debug.print("masked t1: {f}\n", .{t1});
// }

// test "nan inf" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();

//     const allocator = arena.allocator();

//     const arr1 = [5]f32{ 1.0, std.math.inf(f32), std.math.nan(f32), -std.math.inf(f32), -2.3 };

//     // var arr2 = [5]f32{ 1.0, std.math.inf(f32), std.math.nan(f32), -std.math.inf(f32), -2.3 };

//     var t1 = try Self.fromShapedData(allocator, &arr1);

//     // arr1[1] = 2.0;

//     const is_inf = try t1.isInf();
//     const is_pos_inf = try t1.isPositiveInf();
//     const is_neg_inf = try t1.isNegativeInf();
//     const is_nan = try t1.isNan();
//     const is_finite = try t1.isFinite();
//     const is_finite_any = try is_finite.any(null);
//     const is_finite_all = try is_finite.all(null);

//     std.debug.print("t1: {f} is_inf: {f} is_pos_inf: {f} is_neg_inf: {f} is_nan: {f} is_finite: {f}\n", .{ t1, is_inf, is_pos_inf, is_neg_inf, is_nan, is_finite });
//     std.debug.print("is finite: {f} any: {f} all: {f}\n", .{ is_finite, is_finite_any, is_finite_all });

//     const ds = t1.dataSlice(f32);
//     std.debug.print("ds: {any}\n", .{ds});
//     ds[1] = 2.0;

//     try t1.nanToNum_(0.0, .{});
//     std.debug.print("nan_to_num: {f}\n", .{t1});
//     try t1.nanToNum_(0.0, .{ .posinf = 1.0, .neginf = -3.0 });
//     std.debug.print("inf_to_num: {f}\n", .{t1});
// }
