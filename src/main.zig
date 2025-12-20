const std = @import("std");
const tensor = @import("tensor.zig");

const DType = @import("dtype.zig").DataType;
const plot = @import("plot.zig");
const matmul = @import("matmul.zig");

pub fn isStruct(comptime T: type) bool {
    return @typeInfo(T) == .@"struct";
}

fn matmulDemo(allocator: std.mem.Allocator) !void {
    const t1 = try tensor.rand(allocator, [2]usize{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t1: {f} dtype: {}\n", .{ t1.layout, @TypeOf(t1).T });

    var t2 = try tensor.randNorm(allocator, [2]usize{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t2: {f}\n", .{t2.layout});

    t2.transpose_();

    std.debug.print("is contiguous: {}\n", .{t2.isContiguous()});

    const t2_tc = try t2.contiguous();

    std.debug.print("is contiguous: t1= {} t2_tc= {}\n", .{ t1.isContiguous(), t2_tc.isContiguous() });

    const begin = std.time.milliTimestamp();
    const t3 = try matmul.matmul(t1, t2_tc);
    const end = std.time.milliTimestamp();

    std.debug.print("t3: {f}\nelapsed: {d} milliseconds\n", .{ t3.layout, end - begin });
}

pub fn main() !void {
    std.testing.refAllDeclsRecursive(@This());

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    try matmulDemo(allocator);

    // const t1 = try std.Thread.spawn(.{}, generateXY, .{});

    // try plot.beginPlotLoop(allocator);
    // t1.join();
}

fn generateXY() !void {
    var val: f64 = 0.0;
    for (0..1000) |_| {
        const y = @sin(val);
        const y1 = @cos(val);
        try plot.appendData("sin", &.{val}, &.{y});
        try plot.appendData("cos", &.{val}, &.{y1});

        std.posix.nanosleep(0, 10_000_000);

        val += 0.1;
    }
}
