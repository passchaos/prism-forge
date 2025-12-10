const std = @import("std");

pub const DataType = enum {
    f16,
    f32,
    i32,
    u32,

    pub fn typeToDataType(comptime T: type) DataType {
        return switch (T) {
            f16 => .f16,
            f32 => .f32,
            i32 => .i32,
            u32 => .u32,
            else => @compileError("Unsupported type: " ++ @typeName(T)),
        };
    }

    pub fn toType(self: DataType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
            .i32 => i32,
            .u32 => u32,
        };
    }

    pub fn toTypeComp(comptime self: DataType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
            .i32 => i32,
            .u32 => u32,
        };
    }
    pub fn dtypeSize(self: DataType) usize {
        return switch (self) {
            .f16 => 2,
            .f32 => 4,
            .i32 => 4,
            .u32 => 4,
        };
    }
};

pub const Scalar = union(DataType) {
    f16: f16,
    f32: f32,
    i32: i32,
    u32: u32,

    pub fn format(self: @This(), writer: *std.io.Writer) std.Io.Writer.Error!void {
        return switch (self) {
            inline else => |v| writer.print("{d}", .{v}),
        };
    }
};
