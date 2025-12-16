const std = @import("std");
const Tensor = @import("tensor.zig");

const DType = @import("dtype.zig").DataType;
const plot = @import("plot.zig");

pub fn isStruct(comptime T: type) bool {
    return @typeInfo(T) == .@"struct";
}

fn matmulDemo(allocator: std.mem.Allocator) !void {
    const t1 = try Tensor.rand(allocator, f32, &.{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t1: {f}\n", .{t1.layout});

    var t2 = try Tensor.randNorm(allocator, f32, &.{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t2: {f}\n", .{t2.layout});

    const t2_tc = try (try t2.transpose()).contiguous();

    const begin = std.time.milliTimestamp();
    const t3 = try t1.matmul(&t2_tc);
    const end = std.time.milliTimestamp();

    std.debug.print("t3: {f}\nelapsed: {d} milliseconds\n", .{ t3.layout, end - begin });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t1 = try std.Thread.spawn(.{}, generateXY, .{});

    try plot.beginPlotLoop(allocator);
    t1.join();
}

fn generateXY() !void {
    var val: f64 = 0.0;
    for (0..100) |_| {
        const y = @sin(val);
        const y1 = @cos(val);
        try plot.appendData("sin", &.{val}, &.{y});
        try plot.appendData("cos", &.{val}, &.{y1});

        std.posix.nanosleep(0, 100_000_000);

        val += 0.1;
    }
}
