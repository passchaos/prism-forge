const std = @import("std");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;
const DType = tensor.DataType;

pub fn isStruct(comptime T: type) bool {
    return @typeInfo(T) == .@"struct";
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const arr1 = [3][2]f32{
        [2]f32{ 1.0, 2.0 },
        [2]f32{ 3.0, 4.0 },
        [2]f32{ 5.0, 6.0 },
    };
    const t111 = try Tensor.fromShapedData(allocator, &arr1);

    const arr2 = [2][4]f32{
        [4]f32{ 3.0, 4.0, 5.0, 6.0 },
        [4]f32{ 5.0, 6.0, 7.0, 8.0 },
    };
    const t112 = try Tensor.fromShapedData(allocator, &arr2);

    const t113 = try t111.matmul(&t112);
    std.debug.print("t111: {f} t112: {f}\n", .{ t111, t112 });
    std.debug.print("t113: {f}\n", .{t113});
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
