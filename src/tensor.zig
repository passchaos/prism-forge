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

pub fn Tensor(comptime dtype: DataType, comptime DimsTmpl: ?[]const ?usize) type {
    const T = dtype.toType();

    const is_static_rank = DimsTmpl != null;
    const is_all_static = allStatic(DimsTmpl);

    const rank = if (is_static_rank) DimsTmpl.?.len else 0;
    const static_shape = if (is_all_static) blk: {
        var arr: [rank]usize = undefined;
        inline for (DimsTmpl.?, 0..) |dim, i| {
            arr[i] = dim.?;
        }
        break :blk arr;
    } else if (is_static_rank) bmk: {
        break :bmk DimsTmpl.?;
    } else undefined;
    const static_strides = if (is_all_static) blk: {
        var strides: [rank]usize = undefined;

        var acc: usize = 1;
        var i: usize = rank - 1;
        while (i != 0) : (i -= 1) {
            strides[i] = acc;
            acc *= static_shape[i];
        }
        strides[0] = acc;

        break :blk strides;
    } else undefined;

    return struct {
        const Self = @This();

        _shape: if (is_static_rank) [rank]usize else []const usize,
        _strides: if (is_static_rank) [rank]usize else []const usize,
        data: []T,

        pub fn init(allocator: *const std.mem.Allocator, value: T, opts: if (is_all_static) struct {} else if (is_static_rank) struct { shape: [rank]?usize } else struct { shape: []const usize }) if (!is_all_static) anyerror!Self else Self {
            if (is_all_static) {
                var arr = [_]T{value} ** product(&static_shape);

                const a: []T = arr[0..];
                return Self{
                    ._shape = undefined,
                    ._strides = undefined,
                    .data = a,
                };
            } else if (is_static_rank) {
                var dyn_shape: [rank]usize = undefined;

                var len: usize = 1;
                for (static_shape, 0..) |dim, i| {
                    const i_v = if (opts.shape[i]) |v| v else return error.ValueIsNull;
                    const v = if (dim) |val| val else i_v;

                    dyn_shape[i] = v;
                    len *= v;
                }

                var dyn_strides: [rank]usize = undefined;

                var acc: usize = 1;
                var i: usize = rank - 1;
                while (i != 0) : (i -= 1) {
                    dyn_strides[i] = acc;
                    acc *= dyn_shape[i];
                }
                dyn_strides[0] = acc;

                const buf = allocator.alloc(T, len) catch unreachable;
                return Self{
                    ._shape = dyn_shape,
                    ._strides = dyn_strides,
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

        pub fn shape(self: *const Self) if (is_static_rank) [rank]usize else []const usize {
            if (is_all_static) {
                return static_shape;
            } else {
                return self._shape;
            }
        }

        pub fn strides(self: *const Self) if (is_static_rank) [rank]usize else []const usize {
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
                \\}}
            , .{ @typeName(@TypeOf(self)), T, @TypeOf(self.shape()), self.shape(), @TypeOf(self.strides()), self.strides(), self.data.len });
        }
    };
}
