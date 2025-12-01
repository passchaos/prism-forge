const std = @import("std");
const utils = @import("utils.zig");

const asSlice = utils.asSlice;

pub const DataType = enum {
    f32,
    i32,
    u32,

    pub fn toType(self: DataType) type {
        return switch (self) {
            .f32 => f32,
            .i32 => i32,
            .u32 => u32,
        };
    }
};

fn indices_to_flat(indices: []const usize, shape: []const usize, strides: []const usize) anyerror!usize {
    if (indices.len == 0) {
        return error.EmptyIndices;
    }

    var flat_index: usize = 0;
    for (indices, shape, 0..) |index, dim, idx| {
        if (index >= dim) {
            return error.OutOfBounds;
        }

        flat_index += index * strides[idx];
    }
    return flat_index;
}
fn flat_to_indices(flat_index: usize, strides: []const usize) []const usize {
    var indices = [_]usize{0} ** strides.len;
    for (0..strides.len) |dim| {
        indices[dim] = flat_index / strides[dim];
        flat_index %= strides[dim];
    }
    return indices;
}

fn testFloat2Int(v: f32) i32 {
    return @as(i32, @intFromFloat(v));
}

pub fn Tensor(comptime dtype: DataType, comptime DimsTmpl: ?[]const ?usize) type {
    const T = dtype.toType();

    const is_static_rank = DimsTmpl != null;
    const is_all_static = utils.allStatic(DimsTmpl);

    const Rank = if (is_static_rank) DimsTmpl.?.len else 0;
    const static_shape = if (is_all_static) blk: {
        var arr: [Rank]usize = undefined;
        inline for (DimsTmpl.?, 0..) |dim, i| {
            arr[i] = dim.?;
        }
        break :blk arr;
    } else if (is_static_rank) bmk: {
        break :bmk DimsTmpl.?;
    } else undefined;

    return struct {
        const Self = @This();

        const InnerSlice = if (is_static_rank) [Rank]usize else std.ArrayList(usize);
        const DataSlice = std.ArrayList(T);

        _shape: InnerSlice,
        _strides: InnerSlice,
        allocator: std.mem.Allocator,
        data: ?DataSlice,

        // data transform
        pub fn map_i(self: *Self, f: fn (T) T) void {
            if (self.data) |d| {
                for (d.items) |*val| {
                    val.* = f(val.*);
                }
            }
        }

        pub fn map(self: *const Self, allocator: std.mem.Allocator, comptime dtype1: DataType, f: fn (T) dtype1.toType()) anyerror!Tensor(dtype1, DimsTmpl) {
            const data = if (self.data) |d| d else return error.DataIsNull;

            var new_data = try std.ArrayList(dtype1.toType()).initCapacity(allocator, data.items.len);
            try new_data.appendNTimes(allocator, undefined, data.items.len);
            for (data.items, 0..) |val, i| {
                new_data.items[i] = f(val);
            }

            const CopyFn = struct {
                pub inline fn copy(allocator_i: std.mem.Allocator, is: InnerSlice) anyerror!InnerSlice {
                    if (is_static_rank) {
                        return is;
                    } else {
                        return try is.clone(allocator_i);
                    }
                }
            };

            const NewTensorTyp = Tensor(dtype1, DimsTmpl);
            return NewTensorTyp{ ._shape = try CopyFn.copy(allocator, self._shape), ._strides = try CopyFn.copy(allocator, self._strides), .allocator = allocator, .data = new_data };
        }

        // change shape
        pub fn transpose(self: *Self) anyerror!void {
            if (is_static_rank) {
                if (Rank != 2) {
                    @compileError("transpose method can only work with 2-d tensor");
                } else {
                    if (is_all_static) {
                        if (comptime static_shape[0] != static_shape[1]) {
                            @compileError("static shape tensor can only run transpose method with square matrix shape");
                        }
                    }
                }
            } else {
                if (self._shape.items.len != 2) {
                    return error.Non2D;
                }
            }

            if (!is_all_static) {
                if (is_static_rank) {
                    const shape0 = self._shape[0];
                    const shape1 = self._shape[1];

                    self._shape[0] = shape1;
                    self._shape[1] = shape0;
                } else {
                    const shape0 = self._shape.items[0];
                    const shape1 = self._shape.items[1];

                    self._shape.items[0] = shape1;
                    self._shape.items[1] = shape0;
                }
            }

            if (is_static_rank) {
                const stride0 = self._strides[0];
                const stride1 = self._strides[1];
                self._strides[0] = stride1;
                self._strides[1] = stride0;
            } else {
                const stride0 = self._strides.items[0];
                const stride1 = self._strides.items[1];
                self._strides.items[0] = stride1;
                self._strides.items[1] = stride0;
            }
        }

        pub fn reshape(self: *Self, comptime new_shape: []const usize) anyerror!Tensor(dtype, &utils.toOptionalShape(new_shape)) {
            const Typ = Tensor(dtype, &utils.toOptionalShape(new_shape));
            var tensor = try Typ.declare(self.allocator, .{});
            tensor.data = self.data;

            // take self data
            self.data = null;

            return tensor;
        }

        // create method
        pub fn from_data(allocator: std.mem.Allocator, opts: Opts, arr: std.ArrayList(T)) anyerror!Self {
            var tensor = try Self.declare(allocator, opts);
            tensor.data = arr;

            return tensor;
        }

        // construction method
        // only for 2d tensor
        pub fn eye(allocator: std.mem.Allocator, n: usize) anyerror!Self {
            if (is_all_static) {
                @compileError("eye method don't support static shape tensor");
            }

            if (is_static_rank and Rank != 2) {
                @compileError("eye method can only work with 2-d tensor");
            }

            // if (comptime !utils.isNumber(T)) {
            //     @compileError(std.fmt.comptimePrint("eye method can only work with float type, you use: {any}", .{@typeInfo(T)}));
            // }

            const data_len = n * n;
            var arr = try std.ArrayList(T).initCapacity(allocator, data_len);
            try arr.appendNTimes(allocator, 0, data_len);

            for (arr.items) |*elem| elem.* = 0;

            var i: usize = 0;
            while (i < n) : (i += 1) {
                arr.items[i * n + i] = 1;
            }

            if (is_static_rank) {
                return try Self.from_data(allocator, .{ .shape = .{ n, n } }, arr);
            } else {
                var arr_list = try std.ArrayList(usize).initCapacity(allocator, 2);
                try arr_list.appendSlice(allocator, &[2]usize{ n, n });
                return try Self.from_data(allocator, .{ .shape = arr_list }, arr);
            }
        }

        pub fn from_shaped_data(allocator: std.mem.Allocator, arr: anytype) anyerror!Self {
            const info = @typeInfo(@TypeOf(arr));

            const buf: []T = switch (info) {
                .pointer => |ptr| switch (ptr.size) {
                    .one => switch (@typeInfo(ptr.child)) {
                        .array => @as([]T, @ptrCast(@constCast(arr)))[0..],
                        .pointer => |pp| switch (pp.size) {
                            .slice => arr.*,
                            else => @compileError("only support pointer to one"),
                        },
                        else => @compileError(std.fmt.comptimePrint("only support pointer to one: data_type= {any}", .{info})),
                    },
                    else => @compileError("only support pointer to one"),
                },
                else => @compileError("only support pointer to one"),
            };

            if (is_all_static) {
                if (comptime !utils.sliceEqual(&static_shape, utils.getDims(@TypeOf(arr)))) {
                    @compileError(std.fmt.comptimePrint("data shape is mismatched with type: type_shape= {any} data_shape= {any}", .{ static_shape, utils.getDims(@TypeOf(arr)) }));
                }
            }

            if (is_static_rank) {
                if (comptime static_shape.len != utils.getDims(@TypeOf(arr)).len) {
                    @compileError(std.fmt.comptimePrint("data shape len is mismatched with type: type_shape_len= {} data_shape_len= {any}", .{ static_shape.len, utils.getDims(@TypeOf(arr)).len }));
                }
            }

            const dims = utils.getDims(@TypeOf(arr));

            const opts: Opts = if (is_all_static) .{} else if (is_static_rank) .{ .shape = dims } else blk: {
                var arr_list = try std.ArrayList(usize).initCapacity(allocator, dims.len);
                try arr_list.appendSlice(allocator, dims);

                break :blk .{ .shape = arr_list };
            };
            var tensor = try Self.declare(allocator, opts);

            const arr_dims = utils.getDims(@TypeOf(arr));
            if (!std.mem.eql(usize, asSlice(&tensor.shape()), arr_dims)) {
                return anyerror.WrongDimensions;
            }

            var tmp_buf = try std.ArrayList(T).initCapacity(allocator, tensor.size());
            try tmp_buf.appendSlice(allocator, buf);
            tensor.data = tmp_buf;

            return tensor;
        }

        const Opts =
            if (is_all_static) struct {} else if (is_static_rank) struct { shape: [Rank]?usize } else struct { shape: std.ArrayList(usize) };

        pub fn full(allocator: std.mem.Allocator, opts: Opts, value: ?T) anyerror!Self {
            var obj = try Self.declare(allocator, opts);

            const data_len = obj.size();

            var buf = try std.ArrayList(T).initCapacity(allocator, data_len);

            if (value) |v| {
                try buf.appendNTimes(allocator, v, data_len);
            }

            obj.data = buf;

            return obj;
        }

        pub fn arange_count(allocator: std.mem.Allocator, start: T, step: T, count: usize) anyerror!Self {
            if (is_all_static) {
                @compileError("don't support static shape");
            }
            var arr = try std.ArrayList(T).initCapacity(allocator, count);

            var tmp = start;
            for (0..count) |_| {
                try arr.append(allocator, tmp);
                tmp += step;
            }

            // const opts = if (is_static_rank) .{ .shape = [count]} else .{.shape = }
            const opts: Opts = .{ .shape = .{count} };
            var tensor = try Self.declare(allocator, opts);
            tensor.data = arr;

            return tensor;
        }
        pub fn arange_step(allocator: std.mem.Allocator, start: T, end: T, step: T) anyerror!Self {
            if (is_all_static) {
                @compileError("don't support static shape");
            }

            var arr = try std.ArrayList(T).initCapacity(allocator, (end - start) / step);

            var tmp = start;
            while (tmp < end) {
                try arr.append(allocator, tmp);
                tmp += step;
            }

            // const opts = if (is_static_rank) .{ .shape = [count]} else .{.shape = }
            const opts: Opts = .{ .shape = .{arr.items.len} };
            var tensor = try Self.declare(allocator, opts);
            tensor.data = arr;

            return tensor;
        }

        fn declare(allocator: std.mem.Allocator, opts: Opts) anyerror!Self {
            if (is_all_static) {
                var strides_inner: [Rank]usize = undefined;

                var acc: usize = 1;
                var i: usize = Rank - 1;
                while (i != 0) : (i -= 1) {
                    strides_inner[i] = acc;
                    acc *= static_shape[i];
                }
                strides_inner[0] = acc;

                return Self{
                    ._shape = undefined,
                    ._strides = strides_inner,
                    .allocator = allocator,
                    .data = undefined,
                };
            } else if (is_static_rank) {
                var dyn_shape: [Rank]usize = undefined;

                for (static_shape, opts.shape, 0..) |dim, dim_i, i| {
                    if (dim) |v| {
                        if (dim_i) |v_i| {
                            if (v != v_i) return error.DimNotMatch;
                        }

                        dyn_shape[i] = v;
                    } else {
                        if (dim_i) |v_i| {
                            dyn_shape[i] = v_i;
                        } else {
                            return error.MustSpecifyNullDimSize;
                        }
                    }
                }

                var dyn_strides: [Rank]usize = undefined;

                var acc: usize = 1;
                var i: usize = Rank - 1;
                while (i != 0) : (i -= 1) {
                    dyn_strides[i] = acc;
                    acc *= dyn_shape[i];
                }
                dyn_strides[0] = acc;

                return Self{
                    ._shape = dyn_shape,
                    ._strides = dyn_strides,
                    .allocator = allocator,
                    .data = undefined,
                };
            } else {
                const dyn_shape = opts.shape;

                var dyn_strides = try std.ArrayList(usize).initCapacity(allocator, dyn_shape.items.len);
                try dyn_strides.appendNTimes(allocator, 0, dyn_shape.items.len);

                var acc: usize = 1;
                var i: usize = dyn_shape.items.len - 1;
                while (i != 0) : (i -= 1) {
                    dyn_strides.items[i] = acc;
                    acc *= dyn_shape.items[i];
                }
                dyn_strides.items[0] = acc;

                return Self{
                    ._shape = dyn_shape,
                    ._strides = dyn_strides,
                    .allocator = allocator,
                    .data = undefined,
                };
            }
        }

        // core method
        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            if (self.data) |data| {
                @constCast(&data).deinit(allocator);
            }

            if (!is_static_rank) {
                @constCast(&self._shape).deinit(allocator);
                @constCast(&self._strides).deinit(allocator);
            }
        }

        fn get_data_idx(self: *const Self, idx: usize) ?T {
            return if (self.data) |data| data.items[idx] else null;
        }

        pub fn size(self: *const Self) usize {
            var data_len: usize = 1;
            for (self.shape()) |v| {
                data_len *= v;
            }

            return data_len;
        }

        pub fn ndim(self: *const Self) usize {
            if (is_static_rank) {
                return Rank;
            } else {
                return self._shape.items.len;
            }
        }

        pub fn shape(self: *const Self) if (is_static_rank) [Rank]usize else []const usize {
            if (is_static_rank) {
                if (is_all_static) {
                    // use this to prevent shape change
                    return static_shape;
                } else {
                    return self._shape;
                }
            } else {
                return self._shape.items;
            }
        }

        pub fn strides(self: *const Self) if (is_static_rank) [Rank]usize else []const usize {
            if (is_static_rank) {
                return self._strides;
            } else {
                return self._strides.items;
            }
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print(
                \\Tensor{{
                \\  .TypeName = {s}
                \\  .Dtype = {any},
                \\  .Shape = ({any}, {any}),
                \\  .Strides = ({any}, {any}),
                \\  .DataLen = {}
                \\  .Data =
            , .{ @typeName(@TypeOf(self)), T, @TypeOf(self.shape()), self.shape(), @TypeOf(self.strides()), self.strides(), self.size() });

            _ = try writer.write("\n");

            self.fmt_recursive(writer, 0, &.{}) catch |err| {
                std.debug.print("meet failure: {}", .{err});
                return std.Io.Writer.Error.WriteFailed;
            };
            _ = try writer.write("\n}");
        }

        pub fn getIndices(self: *const Self, indices: if (is_static_rank) [Rank]usize else []const usize) !T {
            const flat_index = try indices_to_flat(&indices, &self.shape(), &self.strides());
            return self.data[flat_index];
        }

        fn fmt_recursive(self: *const Self, writer: *std.Io.Writer, depth: usize, indices: []const usize) anyerror!void {
            const dims = self.ndim();

            if (depth == dims) {
                const flat_index = try indices_to_flat(indices, asSlice(&self.shape()), asSlice(&self.strides()));

                try utils.printOptional(writer, "{}", self.get_data_idx(flat_index));
            } else if (depth == dims - 1) {
                try self.fmt_1d_slice(writer, indices);
            } else {
                try self.fmt_nd_slice(writer, depth, indices);
            }
        }

        fn fmt_nd_slice(self: *const Self, writer: *std.Io.Writer, depth: usize, base_indices: []const usize) anyerror!void {
            const pad_show_count = 4;

            const current_dim_size = self.shape()[depth];
            const dims = self.ndim();

            _ = try writer.write("[");

            const show_all = current_dim_size <= 2 * pad_show_count;

            var slice_indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
            defer slice_indices.deinit(self.allocator);

            if (show_all) {
                for (0..current_dim_size) |i| {
                    try slice_indices.append(self.allocator, i);
                }
            } else {
                for (0..pad_show_count) |i| {
                    try slice_indices.append(self.allocator, i);
                }

                for (current_dim_size - pad_show_count..current_dim_size) |i| {
                    try slice_indices.append(self.allocator, i);
                }
            }

            for (slice_indices.items, 0..) |slice_idx, idx| {
                if (idx > 0) {
                    if (depth == dims - 2) {
                        _ = try writer.write("\n ");
                    } else {
                        _ = try writer.write("\n\n ");
                    }

                    for (0..depth) |_| {
                        _ = try writer.write(" ");
                    }
                }

                if (!show_all and idx == pad_show_count) {
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
                }

                var indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
                defer indices.deinit(self.allocator);

                try indices.appendSlice(self.allocator, base_indices);
                try indices.append(self.allocator, slice_idx);

                try self.fmt_recursive(writer, depth + 1, indices.items);
            }

            _ = try writer.write("]");
        }

        fn fmt_1d_slice(self: *const Self, writer: *std.Io.Writer, base_indices: []const usize) anyerror!void {
            const pad_show_count = 5;

            const max_items: usize = if (base_indices.len == 0) 10 else 2 * pad_show_count;
            const current_dim_size = self.shape()[base_indices.len];

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

                    const allocator = self.allocator;
                    var indices = try std.ArrayList(usize).initCapacity(allocator, 4);
                    defer indices.deinit(allocator);

                    try indices.appendSlice(allocator, base_indices);
                    try indices.append(allocator, i);

                    const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));

                    try utils.printOptional(writer, "{}", self.get_data_idx(flat_idx));
                }
            } else {
                for (0..pad_show_count) |i| {
                    if (i > 0) {
                        _ = try writer.write(" ");
                    }

                    var indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
                    defer indices.deinit(self.allocator);

                    try indices.appendSlice(self.allocator, base_indices);
                    try indices.append(self.allocator, i);

                    const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));

                    try utils.printOptional(writer, "{}", self.get_data_idx(flat_idx));
                }
                _ = try writer.write(" ... ");

                for (current_dim_size - pad_show_count..current_dim_size) |i| {
                    var indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
                    defer indices.deinit(self.allocator);

                    try indices.appendSlice(self.allocator, base_indices);
                    try indices.append(self.allocator, i);

                    const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));

                    try utils.printOptional(writer, "{}", self.get_data_idx(flat_idx));

                    if (i < current_dim_size - 1) {
                        _ = try writer.write(" ");
                    }
                }
            }

            _ = try writer.write("]");
        }
    };
}

test "construction test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    const DType = DataType;
    const TensorF32x3x2 = Tensor(DType.f32, &.{ 3, 2 });

    const arr1 = [3][2]f32{
        [2]f32{ 1.0, 2.0 },
        [2]f32{ 3.0, 4.0 },
        [2]f32{ 5.0, 6.0 },
    };
    const t11 = try TensorF32x3x2.from_shaped_data(allocator, &arr1);
    defer t11.deinit(allocator);
    std.debug.print("t11: {f}\n", .{t11});

    const Tensor3U32_1 = Tensor(DType.u32, &.{ 3, null, 5 });
    const t3_1 = try Tensor3U32_1.full(allocator, .{ .shape = .{ 3, 4, 5 } }, 21);
    defer t3_1.deinit(allocator);
    std.debug.print("t3_1: {f}\n", .{t3_1});

    const TensorU32 = Tensor(DType.u32, null);

    var shape4 = try std.ArrayList(usize).initCapacity(allocator, 4);
    try shape4.appendSlice(allocator, &.{ 2, 3, 3, 1, 5 });

    const t4 = try TensorU32.full(allocator, .{ .shape = try shape4.clone(allocator) }, 24);
    defer t4.deinit(allocator);
    std.debug.print("t4: {f}\n", .{t4});

    const t5 = try TensorU32.full(allocator, .{ .shape = shape4 }, 24);
    defer t5.deinit(allocator);
    std.debug.print("t5: {f} {any}\n", .{ t5, t5._shape });

    const Tensor2 = Tensor(DType.f32, &.{ null, null });
    const t6 = try Tensor2.eye(allocator, 10);
    defer t6.deinit(allocator);
    std.debug.print("t6: {f}\n", .{t6});
}

test "arange" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    const Tensor3 = Tensor(DataType.u32, &.{null});

    const t1 = try Tensor3.arange_count(allocator, 1, 2, 20);
    defer t1.deinit(allocator);
    std.debug.print("t1: {f}\n", .{t1});

    const t2 = try Tensor3.arange_step(
        allocator,
        1,
        40,
        2,
    );
    defer t2.deinit(allocator);
    std.debug.print("t2: {f}\n", .{t2});
}

test "shape transform" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    const Tensor112 = Tensor(DataType.u32, &.{ 2, 2 });
    const Tensor122 = Tensor(DataType.u32, &.{ 3, 3 });

    var arr1 = try std.ArrayList(u32).initCapacity(allocator, 5);
    try arr1.appendSlice(allocator, &[_]u32{ 1, 2, 3, 4, 5 });
    var t112 = try Tensor112.from_data(allocator, .{}, arr1);
    defer t112.deinit(allocator);

    try t112.transpose();

    const t112_reshaped = try t112.reshape(&.{ 4, 1 });
    std.debug.print("t112 reshaped: {f}\n", .{t112_reshaped});

    std.debug.print("t112: {f}\n", .{t112});

    const Tensor41 = Tensor(DataType.u32, &.{ 4, 1 });

    var arr1_normal = try std.ArrayList(u32).initCapacity(allocator, 5);
    try arr1_normal.appendSlice(allocator, &[_]u32{ 6, 7, 8, 9, 10 });
    const t112_normal = try Tensor41.from_data(allocator, .{}, arr1_normal);
    defer t112_normal.deinit(allocator);
    std.debug.print("t112 normal: {f}\n", .{t112_normal});

    if (@TypeOf(t112_reshaped) == @TypeOf(t112_normal)) {
        std.debug.print("same type\n", .{});
    } else {
        std.debug.print("different type\n", .{});
    }

    defer t112_reshaped.deinit(allocator);

    var arr2 = try std.ArrayList(u32).initCapacity(allocator, 6);
    try arr2.appendSlice(allocator, &[_]u32{ 1, 2, 3, 4, 5, 6 });
    var t122 = try Tensor122.from_data(allocator, .{}, arr2);
    defer t122.deinit(allocator);

    try t122.transpose();

    const Tensor22 = Tensor(DataType.f32, &.{ null, 5 });
    var t22 = try Tensor22.eye(allocator, 5);
    defer t22.deinit(allocator);

    std.debug.print("t22: {f}\n", .{t22});
    try t22.transpose();
    std.debug.print("t22 transpose: {f}\n", .{t22});

    const Tensor32 = Tensor(DataType.f32, null);
    const t32 = try Tensor32.eye(allocator, 5);
    defer t32.deinit(allocator);

    const arr = [4][5]f32{ [_]f32{ 1, 2, 3, 4, 5 }, [_]f32{ 6, 7, 8, 9, 10 }, [_]f32{ 11, 12, 13, 14, 15 }, [_]f32{ 16, 17, 18, 19, 20 } };
    var t312 = try Tensor32.from_shaped_data(allocator, &arr);
    defer t312.deinit(allocator);

    std.debug.print("t312: {f}\n", .{t312});

    try t312.transpose();

    std.debug.print("t312 transpose: {f}\n", .{t312});
}

test "map related methods" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    // const allocator = gpa.allocator();

    const TensorF32x3x2 = Tensor(DataType.f32, &.{ 3, 2 });

    const arr1 = [3][2]f32{
        [2]f32{ 1.1, 2.2 },
        [2]f32{ 3.3, 4.01 },
        [2]f32{ 5.9, 6.1 },
    };
    var t11 = try TensorF32x3x2.from_shaped_data(allocator, &arr1);
    std.debug.print("t11: {f}\n", .{t11});

    const FnWithCtx = struct {
        pub fn double(x: f32) f32 {
            return 2.0 * x;
        }
        pub fn call(x: f32) i32 {
            return @as(i32, @intFromFloat(x));
        }
    };

    const t11_1 = try t11.map(allocator, DataType.i32, FnWithCtx.call);
    std.debug.print("t11_1: {f}\n", .{t11_1});

    t11.map_i(FnWithCtx.double);
    std.debug.print("t11: {f}\n", .{t11});
}
