const std = @import("std");
const tensor = @import("../tensor.zig");
const log = @import("../log.zig");
const matmul = @import("../matmul.zig");

const DT = f64;
const Tensor2 = tensor.Tensor(2, .{ .T = DT });

pub fn function2(
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

pub const SimpleNet = struct {
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

// if f return !T, compile will hang or meet unnormal compile error
pub fn numericalGradient(
    allocator: std.mem.Allocator,
    comptime N: usize,
    comptime T: type,
    ctx: anytype,
    f: fn (
        comptime N: usize,
        comptime T: type,
        tensor.Tensor(N, .{ .T = T }),
        ctx_f: @TypeOf(ctx),
    ) anyerror!T,
    tval: tensor.Tensor(N, .{ .T = T }),
) !tensor.Tensor(N, .{ .T = T }) {
    const h = 1e-4;

    var tval_v = tval;
    var grad = try tensor.zerosLike(allocator, tval);

    var x_v_iter = tval_v.shapeIter();
    while (x_v_iter.next()) |idx| {
        const tmp_val = try tval_v.getData(idx);

        try tval_v.setData(idx, tmp_val + h);
        const fxh1 = try f(N, T, tval_v, ctx);

        // std.debug.print("tval_v: {f}\n", .{tval_v});
        // std.debug.print("idx: {any} fxh1: {}\n", .{ idx, fxh1 });

        try tval_v.setData(idx, tmp_val - h);
        const fxh2 = try f(N, T, tval_v, ctx);
        // std.debug.print("idx: {any} fxh2: {}\n", .{ idx, fxh2 });

        try grad.setData(idx, (fxh1 - fxh2) / (h + h));

        try tval_v.setData(idx, tmp_val);
    }

    return grad;
}

pub fn gradientDescent(
    allocator: std.mem.Allocator,
    comptime N: usize,
    comptime T: type,
    ctx: anytype,
    f: fn (
        comptime N: usize,
        comptime T: type,
        tensor.Tensor(N, .{ .T = T }),
        ctx_f: @TypeOf(ctx),
    ) anyerror!T,
    init_x: *tensor.Tensor(N, .{ .T = T }),
    args: struct { lr: T = 0.01, step_number: usize = 100 },
) !void {
    for (0..args.step_number) |_| {
        var grad = try numericalGradient(allocator, N, T, ctx, f, init_x.*);
        defer grad.deinit();

        try grad.mul_(args.lr);

        try init_x.sub_(grad);
    }
}

test "differential" {
    const allocator = std.testing.allocator;

    const arr = try tensor.fromArray(allocator, [_]DT{ 3.0, 4.0 });
    defer arr.deinit();

    const v1 = try numericalGradient(allocator, 1, DT, void{}, function2, arr);
    defer v1.deinit();

    log.print(@src(), "v1: {f}\n", .{v1});

    var init_x = try tensor.fromArray(allocator, [_]DT{ -3.0, 4.0 });
    defer init_x.deinit();
    try gradientDescent(allocator, 1, DT, void{}, function2, &init_x, .{ .lr = 0.1 });
    log.print(@src(), "init x: {f}\n", .{init_x});
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

    const result_t = try numericalGradient(allocator, 2, DT, .{ .x = x, .t = t }, tensor_loss, weight);
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
