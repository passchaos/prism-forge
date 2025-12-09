const std = @import("std");
const Tensor = @import("tensor.zig");

const DType = @import("dtype.zig").DataType;

pub fn isStruct(comptime T: type) bool {
    return @typeInfo(T) == .@"struct";
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t1 = try Tensor.rand(allocator, &.{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t1: {f}\n", .{t1.layout});

    const t2 = try Tensor.randNorm(allocator, &.{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t2: {f}\n", .{t2.layout});

    const t2_tc = try (try t2.transpose()).contiguous();

    const begin = std.time.milliTimestamp();
    const t3 = try t1.matmul(&t2_tc);
    const end = std.time.milliTimestamp();

    std.debug.print("t3: {f}\nelapsed: {d} microseconds\n", .{ t3.layout, end - begin });
}
