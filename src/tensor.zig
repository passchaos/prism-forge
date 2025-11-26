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

fn allStatic(comptime dims: []const ?usize) bool {
    inline for (dims) |dim| {
        if (dim == null) return false;
    }
    return true;
}

pub fn Tensor(comptime dtype: DataType, comptime DimsTmpl: []const ?usize) type {
    const T = dtype.toType();
    const is_all_static = allStatic(DimsTmpl);

    const rank = DimsTmpl.len;
    const static_shape: [rank]usize = if (is_all_static) blk: {
        var arr: [rank]usize = undefined;
        inline for (DimsTmpl, 0..) |dim, i| {
            arr[i] = dim.?;
        }
        break :blk arr;
    } else undefined;

    return struct {
        const Self = @This();

        // pub const Rank = DimsTmpl.len;
        // pub const Shape = DimsTmpl;
        _shape: [rank]usize,
        data: []T,

        pub fn init(allocator: *const std.mem.Allocator, value: T, opts: if (is_all_static) struct {} else struct { shape: ?[rank]usize = null }) Self {
            if (is_all_static) {
                // if (opts.shape != null) @compileError("static tensor cannot have shape");
                var arr = [_]T{value} ** product(&static_shape);

                const a: []T = arr[0..];
                return Self{
                    ._shape = undefined,
                    .data = a,
                };
            } else {
                if (opts.shape == null) @panic("no shape gotten");

                const dyn_shape = opts.shape.?;
                if (dyn_shape.len != rank) @panic("shape mismatch");

                var len: usize = 1;
                for (dyn_shape) |dim| {
                    len *= dim;
                }

                const buf = allocator.alloc(T, len) catch unreachable;
                return Self{
                    ._shape = dyn_shape,
                    .data = buf,
                };
            }
        }

        pub fn deinit(self: *const Self, allocator: *const std.mem.Allocator) void {
            if (!is_all_static) {
                allocator.free(self.data);
            }
        }

        pub fn shape(self: *const Self) [rank]usize {
            if (is_all_static) {
                return static_shape;
            } else {
                return self._shape;
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
                \\  .Shape = {any},
                \\  .DataLen = {}
                \\}}
            , .{ @typeName(@TypeOf(self)), T, @TypeOf(self.shape()), self.data.len });
        }
    };
}
