const std = @import("std");
const prism_forge = @import("prism_forge");

pub fn isStruct(comptime T: type) bool {
    return @typeInfo(T) == .@"struct";
}

pub fn main() !void {
    const a = .{ .a = 10, .b = 20 };

    const T = @TypeOf(a);
    inline for (std.meta.fields(T)) |field| {
        std.debug.print("field: {s} type: {any}\n", .{ field.name, field.type });
    }

    // const type_info = comptime @typeInfo(T);
    std.debug.print("a: {} t: {}-{}\n", .{ a, comptime isStruct(T), T });
    // Prints to stderr, ignoring potential errors.
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    try prism_forge.bufferedPrint();
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
