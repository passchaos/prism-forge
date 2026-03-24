pub const DType = enum {
    f32,
    f64,
    i32,
    usize,
    bool,

    const Self = @This();

    pub fn fromAnyType(comptime T: type) Self {
        return switch (T) {
            f32 => .f32,
            f64 => .f64,
            i32 => .i32,
            usize => .usize,
            bool => .bool,
            else => @compileError("Unsupported type"),
        };
    }

    pub fn toType(self: Self) type {
        return switch (self) {
            .f32 => f32,
            .f64 => f64,
            .i32 => i32,
            .usize => usize,
            .bool => bool,
        };
    }

    pub fn dataLen(self: Self) usize {
        return switch (self) {
            .f32 => 4,
            .f64 => 8,
            .i32 => 4,
            .usize => 8,
            .bool => 1,
        };
    }
};
