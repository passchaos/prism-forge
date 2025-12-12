const std = @import("std");

pub const DataType = enum {
    f16,
    f32,
    i32,
    u32,
    bool,

    pub fn typeToDataType(comptime T: type) DataType {
        return switch (T) {
            f16 => .f16,
            f32, comptime_float => .f32,
            i32 => .i32,
            u32, comptime_int => .u32,
            bool => .bool,
            else => @compileError("Unsupported type: " ++ @typeName(T)),
        };
    }

    pub fn toType(self: DataType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
            .i32 => i32,
            .u32 => u32,
            .bool => bool,
        };
    }

    pub fn toTypeComp(comptime self: DataType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
            .i32 => i32,
            .u32 => u32,
            .bool => bool,
        };
    }
    pub fn dtypeSize(self: DataType) usize {
        return switch (self) {
            inline else => |v| @sizeOf(v.toTypeComp()),
        };
    }
};

pub const Scalar = union(DataType) {
    f16: f16,
    f32: f32,
    i32: i32,
    u32: u32,
    bool: bool,

    pub fn from(value: anytype) Scalar {
        return switch (@TypeOf(value)) {
            f16 => .{ .f16 = value },
            f32 => .{ .f32 = value },
            i32 => .{ .i32 = value },
            u32 => .{ .u32 = value },
            bool => .{ .bool = value },
            else => @compileError("Unsupported type: " ++ @typeName(@TypeOf(value))),
        };
    }

    pub fn format(self: @This(), writer: *std.io.Writer) std.Io.Writer.Error!void {
        return switch (self) {
            inline .bool => |v| writer.print("{}", .{v}),
            inline else => |v| writer.print("{d}", .{v}),
        };
    }
};
