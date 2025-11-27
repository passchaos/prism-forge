const std = @import("std");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;
const DType = tensor.DataType;

pub fn isStruct(comptime T: type) bool {
    return @typeInfo(T) == .@"struct";
}

fn fromDataDemo(allocator: std.mem.Allocator) !void {
    const TensorF32x3x2 = Tensor(DType.f32, &.{ 3, 2 });

    const arr1 = [3][2]f32{
        [2]f32{ 1.0, 2.0 },
        [2]f32{ 3.0, 4.0 },
        [2]f32{ 5.0, 6.0 },
    };
    const t11 = try TensorF32x3x2.from_shaped_data(allocator, .{}, &arr1);
    std.debug.print("t11: {f}\n", .{t11});

    const Tensor3U32_1 = Tensor(DType.u32, &.{ 3, null, 5 });
    const t3_1 = try Tensor3U32_1.init(allocator, .{ .shape = .{ 3, 4, 5 } }, 21);
    defer t3_1.deinit(&allocator);
    std.debug.print("t3_1: {f}\n", .{t3_1});

    const TensorU32 = Tensor(DType.u32, null);
    const t4 = try TensorU32.init(allocator, .{ .shape = &.{ 1, 2, 3, 4 } }, 24);
    defer t4.deinit(&allocator);
    std.debug.print("t4: {f}\n", .{t4});

    const t5 = try TensorU32.init(allocator, .{ .shape = &.{ 2, 3, 3, 1, 5 } }, 24);
    defer t5.deinit(&allocator);
    std.debug.print("t5: {f} {any}\n", .{ t5, t5._shape });

    const Tensor2 = Tensor(DType.f32, &.{ null, null });
    const t6 = try Tensor2.eye(allocator, 10);
    defer t6.deinit(&allocator);
    std.debug.print("t6: {f}\n", .{t6});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    try fromDataDemo(allocator);
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
