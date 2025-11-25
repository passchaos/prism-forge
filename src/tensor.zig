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

pub fn Tensor(comptime dtype: DataType, comptime DimsTmpl: []const usize) type {
    const T = dtype.toType();

    return struct {
        const Self = @This();

        pub const Rank = DimsTmpl.len;
        pub const Shape = DimsTmpl;
        data: [product(DimsTmpl)]T,

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
        pub fn init(value: T) Self {
            return Self{
                .data = [_]T{value} ** product(DimsTmpl),
            };
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
            , .{ @typeName(@TypeOf(self)), T, Shape, self.data.len });
        }
    };
}
