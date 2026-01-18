const std = @import("std");
const tensor = @import("../tensor.zig");
const basic = @import("basic.zig");
const layer = @import("layer.zig");
const shape_expr = @import("../shape_expr.zig");

const Affine = layer.Affine;
const Relu = layer.Relu;
const SoftmaxWithLoss = layer.SoftmaxWithLoss;

fn TwoDimDemo(
    comptime batch_size: shape_expr.SizeExpr,
    comptime input_size: shape_expr.SizeExpr,
    comptime hidden_size: shape_expr.SizeExpr,
    comptime output_size: shape_expr.SizeExpr,
) type {
    const Tensor2 = tensor.Tensor(&.{ batch_size, input_size }, f64);

    const Tensor2_L = tensor.Tensor(&.{ batch_size, output_size }, f64);

    const dw1_T = tensor.Tensor(&.{ input_size, hidden_size }, f64);
    const db1_T = tensor.Tensor(&.{ shape_expr.SizeExpr.static(1), hidden_size }, f64);
    const dw2_T = tensor.Tensor(&.{ hidden_size, output_size }, f64);
    const db2_T = tensor.Tensor(&.{ shape_expr.SizeExpr.static(1), output_size }, f64);

    const AffineI = Affine(batch_size, input_size, hidden_size, f64);
    const Affine2I = Affine(batch_size, hidden_size, output_size, f64);
    const ReluI = Relu(&.{ batch_size, hidden_size }, f64);
    const SwlI = SoftmaxWithLoss(&.{ batch_size, output_size }, f64);

    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        // w: Tensor2,
        // b: Tensor2,
        affine: AffineI,
        affine2: Affine2I,
        relu: ReluI,
        last_layer: SwlI,

        fn deinit(self: *Self) void {
            // self..deinit();
            // self.b.deinit();
            self.relu.deinit();
            self.affine.deinit();
            self.affine2.deinit();
            self.last_layer.deinit();
        }

        pub fn init(allocator: std.mem.Allocator, shape_env: *const shape_expr.ShapeEnv) !Self {
            var w = try tensor.randNorm(allocator, &.{ input_size, hidden_size }, shape_env, 0.0, 1.0);
            w.mulScalar_(0.01);
            var b = try tensor.zeros(allocator, f64, &.{ shape_expr.SizeExpr.static(1), hidden_size }, shape_env);
            b.mulScalar_(0.01);

            const w2 = try tensor.randNorm(allocator, &.{ hidden_size, output_size }, shape_env, 0.0, 1.0);
            const b2 = try tensor.zeros(allocator, f64, &.{ shape_expr.SizeExpr.static(1), output_size }, shape_env);

            return Self{
                .allocator = allocator,
                .relu = ReluI.init(),
                .affine = AffineI.init(w, b),
                .affine2 = Affine2I.init(w2, b2),
                .last_layer = SwlI.init(),
            };
        }

        pub fn forward(self: *Self, x: *const Tensor2) !Tensor2_L {
            const f1 = try self.affine.forward(x);
            defer f1.deinit();
            const f2 = try self.relu.forward(&f1);
            defer f2.deinit();
            const f3 = try self.affine2.forward(&f2);
            // var f1 = try x.matmul(&self.w);

            // try f1.add_(&self.b);

            // try f1.relu_();

            return f3;
        }

        pub fn loss(self: *Self, x: *const Tensor2, t: *const Tensor2_L) !f64 {
            var f1 = try self.forward(x);
            defer f1.deinit();

            const res = try self.last_layer.forward(&f1, t);
            return res;
        }

        pub fn backward_orig(self: *Self, x: *const Tensor2, t: *const Tensor2_L) !struct {
            dw: dw1_T,
            db: db1_T,
            dw2: dw2_T,
            db2: db2_T,
        } {
            _ = try self.loss(x, t);
            // defer logits.deinit();

            const g1 = try self.last_layer.backward();
            defer g1.deinit();

            const a2_g = try self.affine2.backward(&g1);
            defer a2_g.deinit();

            const r1 = try self.relu.backward(&a2_g);
            defer r1.deinit();

            const g2 = try self.affine.backward(&r1);
            defer g2.deinit();

            return .{
                .dw = try self.affine.dw.?.clone(),
                .db = try self.affine.db.?.clone(),
                .dw2 = try self.affine2.dw.?.clone(),
                .db2 = try self.affine2.db.?.clone(),
            };
        }

        pub fn backward(self: *Self, x: *const Tensor2, t: *const Tensor2_L) !struct { dw: Tensor2, db: Tensor2 } {
            const logits = try self.forward(x);
            defer logits.deinit();

            var logits_softmax = try logits.softmax();
            defer logits_softmax.deinit();

            try logits_softmax.sub_(t);
            try logits_softmax.div_(@as(f64, @floatFromInt(batch_size)));

            const x_t = x.transpose();
            defer x_t.deinit();

            // const dw = try x_t.matmul(&logits_softmax);
            // const db = try logits_softmax.sum(0);

            // const w2_t = self.affine2.w.transpose();
            // defer w2_t.deinit();
            // var da1 = try logits_softmax.matmul(&w2_t);
            // defer da1.deinit();

            // try da1.maskFill_(self.relu.mask.?, 0.0);
            // try da1.mul_(self.relu.mask.?);

            const dw1 = try x_t.matmul(&logits_softmax);
            const db1 = try logits_softmax.sum(0);

            return .{
                .dw = dw1,
                .db = db1,
            };
        }

        pub fn numericalGradient(self: *Self, x: *const Tensor2, t: *const Tensor2_L) !struct { dw: Tensor2, db: Tensor2, dw2: Tensor2, db2: Tensor2 } {
            const dw = try basic.numericalGradient(self.allocator, self, basic.LossArgument{
                .x = x,
                .t = t,
            }, basic.net_loss, &self.affine.w);

            const db = try basic.numericalGradient(self.allocator, self, basic.LossArgument{
                .x = x,
                .t = t,
            }, basic.net_loss, &self.affine.b);

            const dw2 = try basic.numericalGradient(self.allocator, self, basic.LossArgument{
                .x = x,
                .t = t,
            }, basic.net_loss, &self.affine2.w);

            const db2 = try basic.numericalGradient(self.allocator, self, basic.LossArgument{
                .x = x,
                .t = t,
            }, basic.net_loss, &self.affine2.b);

            return .{
                .dw = dw,
                .db = db,
                .dw2 = dw2,
                .db2 = db2,
            };
        }
    };
}

test "numerical and analytic gradients" {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();

    // const allocator = gpa.allocator();

    // var arena = std.heap.ArenaAllocator.init(g_allocator);
    // defer arena.deinit();
    // const allocator = arena.allocator();
    const allocator = std.testing.allocator;

    const input_dim = comptime shape_expr.SizeExpr.static(784);
    const hidden_dim = comptime shape_expr.SizeExpr.static(50);
    const output_dim = comptime shape_expr.SizeExpr.static(10);
    const batch_size = comptime shape_expr.SizeExpr.static(50);

    var shape_env = shape_expr.ShapeEnv.init(allocator);
    defer shape_env.deinit();

    const x = try tensor.randNorm(allocator, &.{ batch_size, input_dim }, &shape_env, 0.0, 1.0);
    defer x.deinit();

    const ta = try tensor.rand(allocator, &.{batch_size}, &shape_env, 0, 10);
    defer ta.deinit();
    const t = try ta.oneHot(f64, 10);
    std.debug.print("t: {f}\n", .{t});
    // const t = try tensor.fromArray(allocator, [batch_size][output_dim]f64{
    //     .{ 0.0, 0.0, 1.0, 0.0, 0.0 },
    //     .{ 1.0, 0.0, 0.0, 0.0, 0.0 },
    // });
    defer t.deinit();

    const Net = TwoDimDemo(
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
    );

    var net = try Net.init(allocator, &shape_env);
    defer net.deinit();

    var analytic_orig_grads = try net.backward_orig(&x, &t);
    defer {
        analytic_orig_grads.dw.deinit();
        analytic_orig_grads.db.deinit();
    }
    // const analytic_grads = try net.backward(&x, &t);
    // defer {
    //     analytic_grads.dw.deinit();
    //     analytic_grads.db.deinit();
    // }
    const numerical_grads = try net.numericalGradient(&x, &t);
    defer {
        numerical_grads.dw.deinit();
        numerical_grads.db.deinit();
    }

    // try analytic_orig_grads.dw.sub_(&analytic_grads.dw);
    // try analytic_orig_grads.db.sub_(&analytic_grads.db);
    try analytic_orig_grads.dw.sub_(&numerical_grads.dw);
    try analytic_orig_grads.db.sub_(&numerical_grads.db);
    try analytic_orig_grads.dw2.sub_(&numerical_grads.dw2);
    try analytic_orig_grads.db2.sub_(&numerical_grads.db2);

    analytic_orig_grads.dw.abs_();
    analytic_orig_grads.db.abs_();
    analytic_orig_grads.dw2.abs_();
    analytic_orig_grads.db2.abs_();

    const diff_dw = try analytic_orig_grads.dw.meanAll();
    defer diff_dw.deinit();
    const diff_db = try analytic_orig_grads.db.meanAll();
    defer diff_db.deinit();
    const diff_dw2 = try analytic_orig_grads.dw2.meanAll();
    defer diff_dw2.deinit();
    const diff_db2 = try analytic_orig_grads.db2.meanAll();
    defer diff_db2.deinit();

    const diff_dw_v = try diff_dw.dataItem();
    const diff_db_v = try diff_db.dataItem();
    const diff_dw2_v = try diff_dw2.dataItem();
    const diff_db2_v = try diff_db2.dataItem();

    const TV = 1e-10;
    try std.testing.expect(diff_dw_v < TV);
    try std.testing.expect(diff_db_v < TV);
    try std.testing.expect(diff_dw2_v < TV);
    try std.testing.expect(diff_db2_v < TV);

    std.debug.print("diff: dw= {} db= {} dw2= {} db2= {}\n", .{ diff_dw_v, diff_db_v, diff_dw2_v, diff_db2_v });
}
