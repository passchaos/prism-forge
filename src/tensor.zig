const std = @import("std");

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

fn toArray(comptime slice: []const usize) [slice.len]usize {
    var result: [slice.len]usize = undefined;
    inline for (slice, 0..) |dim, i| {
        result[i] = dim;
    }
    return result;
}

fn ElemOf(comptime V: type) type {
    return switch (@typeInfo(V)) {
        .pointer => |info| switch (info.size) {
            .slice => info.child,
            .one => switch (@typeInfo(info.child)) {
                .array => |arr| arr.child,
                .pointer => |p| p.child,
                else => |v| @compileError(std.fmt.comptimePrint("Unsupported pointer type: info= {} info_child= {}\n", .{ info, v })),
            },
            else => @compileError("Unsupported pointer type"),
        },
        .array => @compileError("ElementOf: use array will get a copy of argument, so can't get valid value"),
        else => @compileError("Unsupported type"),
    };
}

pub fn asSlice(value: anytype) []const ElemOf(@TypeOf(value)) {
    const tito = @typeInfo(@TypeOf(value));
    std.debug.print("Type info: {any}\n", .{tito});

    return switch (@typeInfo(@TypeOf(value))) {
        .array => @compileError("asSlice: use array will get a copy of argument, so can't get valid value"),
        .pointer => |p| switch (p.size) {
            .one => switch (@typeInfo(p.child)) {
                .array => value,
                .slice => *value,
                else => |v| @compileError(std.fmt.comptimePrint("unsupported pointer to non-array: {}", .{v})),
            },
            .slice => value,
            else => @compileError("Unsupported pointer type"),
        },
        else => @compileError("Unsupported type"),
    };
}

fn product(comptime arr: []const usize) usize {
    var result: usize = 1;
    inline for (arr) |dim| {
        result *= dim;
    }
    return result;
}

fn allStatic(comptime dims: ?[]const ?usize) bool {
    if (dims == null) return false;

    inline for (dims.?) |dim| {
        if (dim == null) return false;
    }
    return true;
}

fn indices_to_flat(indices: []const usize, shape: []const usize, strides: []const usize) anyerror!usize {
    if (indices.len == 0) {
        return error.EmptyIndices;
    }

    std.debug.print("indices= {any} shape= {any} strides= {any}\n", .{ indices, shape, strides });
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

pub fn Tensor(comptime dtype: DataType, comptime DimsTmpl: ?[]const ?usize) type {
    const T = dtype.toType();

    const is_static_rank = DimsTmpl != null;
    const is_all_static = allStatic(DimsTmpl);

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
    const static_strides = if (is_all_static) blk: {
        var strides: [Rank]usize = undefined;

        var acc: usize = 1;
        var i: usize = Rank - 1;
        while (i != 0) : (i -= 1) {
            strides[i] = acc;
            acc *= static_shape[i];
        }
        strides[0] = acc;

        break :blk strides;
    } else undefined;

    return struct {
        const Self = @This();

        _shape: if (is_static_rank) [Rank]usize else []const usize,
        _strides: if (is_static_rank) [Rank]usize else []const usize,
        allocator: std.mem.Allocator,
        data: []T,

        pub fn init(allocator: std.mem.Allocator, value: T, opts: if (is_all_static) struct {} else if (is_static_rank) struct { shape: [Rank]?usize } else struct { shape: []const usize }) if (!is_all_static) anyerror!Self else Self {
            if (is_all_static) {
                var arr = [_]T{value} ** product(&static_shape);

                const a: []T = arr[0..];
                return Self{
                    ._shape = undefined,
                    ._strides = undefined,
                    .allocator = allocator,
                    .data = a,
                };
            } else if (is_static_rank) {
                var dyn_shape: [Rank]usize = undefined;

                var len: usize = 1;
                for (static_shape, 0..) |dim, i| {
                    const i_v = if (opts.shape[i]) |v| v else return error.ValueIsNull;
                    const v = if (dim) |val| val else i_v;

                    dyn_shape[i] = v;
                    len *= v;
                }

                var dyn_strides: [Rank]usize = undefined;

                var acc: usize = 1;
                var i: usize = Rank - 1;
                while (i != 0) : (i -= 1) {
                    dyn_strides[i] = acc;
                    acc *= dyn_shape[i];
                }
                dyn_strides[0] = acc;

                const buf = allocator.alloc(T, len) catch unreachable;
                return Self{
                    ._shape = dyn_shape,
                    ._strides = dyn_strides,
                    .allocator = allocator,
                    .data = buf,
                };
            } else {
                const dyn_shape = opts.shape;

                var len: usize = 1;
                for (dyn_shape) |dim| {
                    len *= dim;
                }

                var dyn_strides: []usize = try allocator.alloc(usize, dyn_shape.len);

                var acc: usize = 1;
                var i: usize = dyn_shape.len - 1;
                while (i != 0) : (i -= 1) {
                    dyn_strides[i] = acc;
                    acc *= dyn_shape[i];
                }
                dyn_strides[0] = acc;

                const buf = try allocator.alloc(T, len);
                return Self{
                    ._shape = dyn_shape,
                    ._strides = dyn_strides,
                    .allocator = allocator,
                    .data = buf,
                };
            }
        }

        pub fn deinit(self: *const Self, allocator: *const std.mem.Allocator) void {
            if (!is_all_static) {
                allocator.free(self.data);
            }

            if (!is_static_rank) {
                allocator.free(self._strides);
            }
        }

        pub fn shape(self: *const Self) if (is_static_rank) [Rank]usize else []const usize {
            if (is_all_static) {
                return static_shape;
            } else {
                return self._shape;
            }
        }

        pub fn strides(self: *const Self) if (is_static_rank) [Rank]usize else []const usize {
            if (is_all_static) {
                return static_strides;
            } else {
                return self._strides;
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
            , .{ @typeName(@TypeOf(self)), T, @TypeOf(self.shape()), self.shape(), @TypeOf(self.strides()), self.strides(), self.data.len });

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
            const ndim = self.shape().len;

            if (depth == ndim) {
                const flat_index = try indices_to_flat(indices, asSlice(&self.shape()), asSlice(&self.strides()));
                try writer.print("{d}", .{self.data[flat_index]});
            } else if (depth == ndim - 1) {
                try self.fmt_1d_slice(writer, indices);
            } else {
                try self.fmt_nd_slice(writer, depth, indices);
            }
        }

        fn fmt_nd_slice(self: *const Self, writer: *std.Io.Writer, depth: usize, base_indices: []const usize) anyerror!void {
            const pad_show_count = 8;

            const current_dim_size = self.shape()[depth];
            const ndim = self.shape().len;

            const show_all = current_dim_size <= 2 * pad_show_count;

            var slice_indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);

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
                    if (depth == ndim - 2) {
                        _ = try writer.write("\n ");
                    } else {
                        _ = try writer.write("\n\n ");
                    }

                    for (0..depth) |_| {
                        _ = try writer.write(" ");
                    }
                }

                if (show_all and idx == pad_show_count) {
                    if (depth == ndim - 2) {
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
                try indices.appendSlice(self.allocator, base_indices);
                try indices.append(self.allocator, slice_idx);

                try self.fmt_recursive(writer, depth + 1, indices.items);
            }

            _ = try writer.write("]");
        }

        fn fmt_1d_slice(self: *const Self, writer: *std.Io.Writer, base_indices: []const usize) anyerror!void {
            const pad_show_count = 5;

            const max_items: usize = if (base_indices.len == 0) 1000 else 2 * pad_show_count;
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
                    try indices.appendSlice(allocator, base_indices);
                    try indices.append(allocator, i);

                    const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));
                    try writer.print("{}", .{self.data[flat_idx]});
                }
            } else {
                for (0..pad_show_count) |i| {
                    if (i > 0) {
                        _ = try writer.write(" ");
                    }

                    var indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
                    try indices.appendSlice(self.allocator, base_indices);
                    try indices.append(self.allocator, i);

                    const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));
                    try writer.print("{}", .{self.data[flat_idx]});
                }
                _ = try writer.write(" ... ");

                for (current_dim_size - pad_show_count..current_dim_size) |i| {
                    var indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
                    try indices.appendSlice(self.allocator, base_indices);
                    try indices.append(self.allocator, i);

                    const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));
                    try writer.print("{}", .{self.data[flat_idx]});
                    if (i < current_dim_size - 1) {
                        _ = try writer.write(" ");
                    }
                }
            }

            _ = try writer.write("]");
        }
    };
}
