const std = @import("std");
const tensor = @import("tensor.zig");

const DType = @import("dtype.zig").DataType;
const plot = @import("plot.zig");
const basic = @import("nn/basic.zig");
const layer = @import("nn/layer.zig");
const log = @import("log.zig");
const tools = @import("nn/tools.zig");

const mnist_train = @import("demo/mnist_train.zig");

pub fn logFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = scope;

    const src = @src();
    log.print(@src(), "{s}:{d} [{s}] " ++ format ++ "\n", .{
        src.file, src.line, @tagName(level),
    } ++ args);
    // std.log.defaultLog(level, scope, format, args);
}

pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = logFn,
};

pub fn isStruct(comptime T: type) bool {
    return @typeInfo(T) == .@"struct";
}

fn matmulDemo(allocator: std.mem.Allocator) !void {
    const t1 = try tensor.rand(allocator, [2]usize{ 3000, 3000 }, 0.0, 1.0);
    log.print(@src(), "t1: {f} dtype: {}\n", .{ t1.layout, @TypeOf(t1).T });

    var t2 = try tensor.randNorm(allocator, [2]usize{ 3000, 3000 }, 0.0, 1.0);
    log.print(@src(), "t2: {f}\n", .{t2.layout});

    t2.transpose_();

    log.print(@src(), "is contiguous: {}\n", .{t2.isContiguous()});

    const t2_tc = try t2.contiguous();

    log.print(@src(), "is contiguous: t1= {} t2_tc= {}\n", .{ t1.isContiguous(), t2_tc.isContiguous() });
}

pub fn main() !void {
    // try layer.testNumericalAndAnalyticGrad();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    // // try matmulDemo(allocator);

    // const t1 = try std.Thread.spawn(.{}, basic.twoLayerNetTrain, .{ allocator, 1000, 100, 0.01 });

    const t1 = try std.Thread.spawn(
        .{},
        mnist_train.trainNet,
        .{ allocator, 1000, 100, 0.01 },
    );
    try plot.beginPlotLoop(allocator);
    t1.join();
    log.print(@src(), "finish main logic\n", .{});
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

test {
    const demo = @import("nn/demo.zig");

    std.testing.refAllDeclsRecursive(@This());
    std.testing.refAllDeclsRecursive(basic);
    std.testing.refAllDeclsRecursive(demo);
    std.testing.refAllDecls(tools);
}
