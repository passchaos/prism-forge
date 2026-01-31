const std = @import("std");
const tensor = @import("../tensor.zig");
const log = @import("../log.zig");
const mnist = @import("../mnist.zig");
const plot = @import("../plot.zig");
const layer = @import("layer.zig");
const shape_expr = @import("../shape_expr.zig");
const optim = @import("optim.zig");

const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;
const Layer = layer.Layer;

const DT = f64;
fn Tensor2(comptime shape_expa_a: [2]SizeExpr) type {
    return tensor.Tensor(&shape_expa_a, DT);
}

pub fn function2(
    comptime shape_expr_a: [2]usize,
    input: Tensor2(shape_expr_a),
    _: void,
) anyerror!DT {
    var input_iter = input.shapeIter();

    var result: DT = 0;

    while (input_iter.next()) |idx| {
        result += std.math.pow(DT, try input.getData(idx), 2);
    }

    return result;
}

pub fn LossArgument(comptime shape_expr_a: [2]SizeExpr) type {
    return struct {
        x: *const Tensor2(shape_expr_a),
        t: *const Tensor2(shape_expr_a),
    };
}

pub fn net_loss(
    net: anytype,
    ctx: anytype,
) anyerror!DT {
    const loss = try net.loss(ctx.x, ctx.t);

    return loss;
}

pub fn SimpleNet(comptime batch_size: SizeExpr, comptime shape_expr_a: [2]SizeExpr) type {
    const Tensor = Tensor2(shape_expr_a);
    return struct {
        w: Tensor,

        const Self = @This();

        fn deinit(self: *const Self) void {
            return self.w.deinit();
        }

        fn resetWeight(self: *Self, new_w: Tensor) void {
            self.w.deinit();
            self.w = new_w;
        }

        fn init(weight: Tensor) Self {
            return Self{ .w = weight };
        }

        fn predict(
            self: *const Self,
            x: *const Tensor2([2]SizeExpr{ batch_size, shape_expr_a[0] }),
        ) !Tensor2([2]SizeExpr{ batch_size, shape_expr_a[1] }) {
            return try x.matmul(&self.w);
        }

        fn loss(
            self: *const Self,
            x: *const Tensor2([2]SizeExpr{ batch_size, shape_expr_a[0] }),
            t: *const Tensor2([2]SizeExpr{ batch_size, shape_expr_a[1] }),
        ) !DT {
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
}

// if f return !T, compile will hang or meet unnormal compile error
pub fn numericalGradient(
    allocator: std.mem.Allocator,
    comptime shape_expr_a: []const SizeExpr,
    net: anytype,
    ctx: anytype,
    f: fn (
        anytype,
        anytype,
    ) anyerror!DT,
    tval: *tensor.Tensor(shape_expr_a, DT),
) !tensor.Tensor(shape_expr_a, DT) {
    const h = 1e-5;

    var grad = try tensor.zerosLike(allocator, tval.*);

    var x_v_iter = tval.shapeIter();
    while (x_v_iter.next()) |idx| {
        const tmp_val = try tval.getData(idx);

        try tval.setData(idx, tmp_val + h);
        const fxh1 = try f(net, ctx);

        // std.debug.print("tval_v: {f}\n", .{tval_v});
        // std.debug.print("idx: {any} fxh1: {}\n", .{ idx, fxh1 });

        try tval.setData(idx, tmp_val - h);
        const fxh2 = try f(net, ctx);
        // std.debug.print("idx: {any} fxh2: {}\n", .{ idx, fxh2 });

        try grad.setData(idx, (fxh1 - fxh2) / (h + h));

        try tval.setData(idx, tmp_val);
    }

    return grad;
}

pub fn gradientDescent(
    allocator: std.mem.Allocator,
    comptime shape: [2]usize,
    ctx: anytype,
    f: fn (
        Tensor2(shape),
        ctx_f: @TypeOf(ctx),
    ) anyerror!DT,
    init_x: *Tensor2(shape),
    args: struct { lr: DT = 0.01, step_number: usize = 100 },
) !void {
    for (0..args.step_number) |_| {
        var grad = try numericalGradient(allocator, ctx, f, init_x.*);
        defer grad.deinit();

        try grad.mul_(args.lr);

        try init_x.sub_(grad);
    }
}

test "simple net" {
    const allocator = std.testing.allocator;

    const weight = try tensor.fromArray(allocator, [_][3]DT{
        .{ 0.47355232, 0.9977393, 0.84668094 },
        .{ 0.85557411, 0.03563661, 0.69422093 },
    });

    const shape_expr_a =
        comptime [2]SizeExpr{ SizeExpr.static(2), SizeExpr.static(3) };
    var net = SimpleNet(SizeExpr.static(1), shape_expr_a).init(weight);
    defer net.deinit();

    log.print(@src(), "w: {f}\n", .{net.w});

    const x = try tensor.fromArray(allocator, [_][2]DT{
        .{ 0.6, 0.9 },
    });
    defer x.deinit();

    const t = try tensor.fromArray(allocator, [_][3]DT{
        .{ 0.0, 0.0, 1.0 },
    });
    defer t.deinit();

    const loss = try net.loss(&x, &t);
    try std.testing.expectApproxEqAbs(0.9280682857864075, loss, 1e-5);
    log.print(@src(), "loss: {}\n", .{loss});

    const result_t = try numericalGradient(
        allocator,
        shape_expr_a,
        &net,
        .{ .x = &x, .t = &t },
        net_loss,
        &net.w,
    );
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
