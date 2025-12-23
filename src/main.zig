const std = @import("std");
const tensor = @import("tensor.zig");

const DType = @import("dtype.zig").DataType;
const plot = @import("plot.zig");
const matmul = @import("matmul.zig");
const basic = @import("nn/basic.zig");
const log = @import("log.zig");

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

    const begin = std.time.milliTimestamp();
    const t3 = try matmul.matmul(t1, t2_tc);
    const end = std.time.milliTimestamp();

    log.print(@src(), "t3: {f}\nelapsed: {d} milliseconds\n", .{ t3.layout, end - begin });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    // try matmulDemo(allocator);
    const t1 = try tensor.rand(allocator, [2]usize{ 3, 5 }, 2.0, 10.0);
    const t2 = try tensor.rand(allocator, [2]usize{ 3, 5 }, 3.0, 9.0);

    const t1am = try t1.argMax(1);
    const t2am = try t2.argMin(1);

    const t3 = try (try t1am.eql(t2am)).sumAll();

    log.print(@src(), "t1am: {f} t2am: {f} t3: {f}\n", .{ t1am, t2am, t3 });

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

const DT = f64;
fn function2(
    comptime N: usize,
    comptime T: type,
    input: tensor.Tensor(N, .{ .T = T }),
    _: void,
) anyerror!T {
    var input_iter = input.shapeIter();

    var result: T = 0;

    while (input_iter.next()) |idx| {
        result += std.math.pow(T, try input.getData(idx), 2);
    }

    return result;
}

test "differential" {
    const allocator = std.testing.allocator;

    const arr = try tensor.fromArray(allocator, [_]DT{ 3.0, 4.0 });
    defer arr.deinit();

    const v1 = try basic.numericalGradient(allocator, 1, DT, void{}, function2, arr);
    defer v1.deinit();

    log.print(@src(), "v1: {f}\n", .{v1});

    var init_x = try tensor.fromArray(allocator, [_]DT{ -3.0, 4.0 });
    defer init_x.deinit();
    try basic.gradientDescent(allocator, 1, DT, void{}, function2, &init_x, .{ .lr = 0.1 });
    log.print(@src(), "init x: {f}\n", .{init_x});
}

pub const Tensor2 = tensor.Tensor(2, .{ .T = DT });
const SimpleNet = struct {
    w: Tensor2,

    const Self = @This();

    // fn deinit(self: *const Self) void {
    //     return self.w.deinit();
    // }

    fn resetWeight(self: *Self, new_w: Tensor2) void {
        self.w.deinit();
        self.w = new_w;
    }

    fn init(weight: Tensor2) Self {
        return Self{ .w = weight };
    }

    fn predict(self: *const Self, x: Tensor2) !Tensor2 {
        return try matmul.matmul(x, self.w);
    }

    fn loss(self: *const Self, x: Tensor2, t: Tensor2) !DT {
        const z = try self.predict(x);
        // log.print(@src(), "z: {f}\n", .{z});
        defer z.deinit();
        const y = try z.softmax();
        defer y.deinit();

        // log.print(@src(), "y: {f}\n", .{y});

        const cross_entropy = try y.crossEntropy(t);
        defer cross_entropy.deinit();
        return try cross_entropy.dataItem();
    }
};

fn tensor_loss(
    comptime N: usize,
    comptime T: type,
    input: tensor.Tensor(N, .{ .T = T }),
    ctx: struct {
        x: tensor.Tensor(N, .{ .T = T }),
        t: tensor.Tensor(N, .{ .T = T }),
    },
) anyerror!T {
    const net = SimpleNet.init(input);

    const loss = try net.loss(ctx.x, ctx.t);

    return loss;
}

test "simple net" {
    const allocator = std.testing.allocator;

    const weight = try tensor.fromArray(allocator, [_][3]DT{
        .{ 0.47355232, 0.9977393, 0.84668094 },
        .{ 0.85557411, 0.03563661, 0.69422093 },
    });
    defer weight.deinit();

    var net = SimpleNet.init(weight);
    // defer net.deinit();

    log.print(@src(), "w: {f}\n", .{net.w});

    const x = try tensor.fromArray(allocator, [_][2]DT{
        .{ 0.6, 0.9 },
    });
    defer x.deinit();

    const t = try tensor.fromArray(allocator, [_][3]DT{
        .{ 0.0, 0.0, 1.0 },
    });
    defer t.deinit();

    const loss = try net.loss(x, t);
    try std.testing.expectApproxEqAbs(0.9280682857864075, loss, 1e-15);
    log.print(@src(), "loss: {}\n", .{loss});

    const result_t = try basic.numericalGradient(allocator, 2, DT, .{ .x = x, .t = t }, tensor_loss, weight);
    // _ = result_t;
    defer result_t.deinit();

    const expected_result = try tensor.fromArray(allocator, [_][3]DT{
        .{ 0.21924763, 0.14356247, -0.36281009 },
        .{ 0.32887144, 0.2153437, -0.54421514 },
    });
    defer expected_result.deinit();

    const approx_eq = result_t.approxEqual(expected_result, 1e-6, 1e-7);
    try std.testing.expect(approx_eq);

    std.debug.print("result_t: {f}\n", .{result_t});
}
