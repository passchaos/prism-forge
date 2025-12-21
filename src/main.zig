const std = @import("std");
const tensor = @import("tensor.zig");

const DType = @import("dtype.zig").DataType;
const plot = @import("plot.zig");
const matmul = @import("matmul.zig");
const basic = @import("nn/basic.zig");

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

fn function2(
    comptime N: usize,
    input: tensor.Tensor(N, .{}),
) f32 {
    var input_iter = input.shapeIter();

    var result: f32 = 0;

    while (input_iter.next()) |idx| {
        result += std.math.pow(f32, input.getData(idx) catch unreachable, 2);
    }

    return result;
}

test "differential" {
    const allocator = std.testing.allocator;

    const arr = try tensor.fromArray(allocator, [_]f32{ 3.0, 4.0 });
    defer arr.deinit();

    const v1 = try basic.numericalGradient(allocator, 1, function2, arr);
    defer v1.deinit();

    std.debug.print("v1: {f}\n", .{v1});

    var init_x = try tensor.fromArray(allocator, [_]f32{ -3.0, 4.0 });
    defer init_x.deinit();
    try basic.gradientDescent(allocator, 1, function2, &init_x, .{ .lr = 0.1 });
    std.debug.print("init x: {f}\n", .{init_x});
}

pub const Tensor2 = tensor.Tensor(2, .{});
const SimpleNet = struct {
    w: Tensor2,

    const Self = @This();

    fn deinit(self: *const Self) void {
        return self.w.deinit();
    }

    fn init(allocator: std.mem.Allocator) !Self {
        const w = try tensor.randNorm(allocator, [2]usize{ 2, 3 }, 0.0, 1.0);
        return Self{ .w = w };
    }

    fn predict(self: *const Self, x: Tensor2) !Tensor2 {
        return try matmul.matmul(x, self.w);
    }

    fn loss(self: *const Self, x: Tensor2, t: Tensor2) !f32 {
        const z = try self.predict(x);
        defer z.deinit();
        const y = try z.softmax();
        defer y.deinit();

        const cross_entropy = try y.crossEntropy(t);
        defer cross_entropy.deinit();
        return try cross_entropy.dataItem();
    }
};

test "simple net" {
    const allocator = std.testing.allocator;

    const net = try SimpleNet.init(allocator);
    defer net.deinit();

    std.debug.print("w: {f}\n", .{net.w});

    const x = try tensor.fromArray(allocator, [_][2]f32{
        .{ 0.6, 0.9 },
    });
    defer x.deinit();

    const t = try tensor.fromArray(allocator, [_][3]f32{
        .{ 0.0, 0.0, 1.0 },
    });
    defer t.deinit();

    const loss = try net.loss(x, t);
    std.debug.print("loss: {}\n", .{loss});
}
