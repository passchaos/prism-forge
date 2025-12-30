const std = @import("std");
const tensor = @import("../tensor.zig");
const log = @import("../log.zig");
const basic = @import("basic.zig");

pub fn Relu(comptime shape: []const usize, comptime T: type) type {
    const Tensor = tensor.Tensor(shape, T, .{});
    const BoolTensor = tensor.Tensor(shape, bool, .{});

    return struct {
        mask: ?BoolTensor = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.mask.?.deinit();
        }

        pub fn init() Self {
            return Self{};
        }

        pub fn forward(self: *Self, x: *const Tensor) !Tensor {
            if (self.mask) |m_i| {
                m_i.deinit();
            }
            self.mask = try x.leScalar(0);

            var nr = try x.clone();
            // log.print(@src(), "mask layout: {f}\n", .{self.mask.?.layout});

            try nr.maskFill_(self.mask.?, @as(T, 0));

            return nr;
        }

        pub fn backward(self: *Self, dout: *const Tensor) !Tensor {
            var res = try dout.clone();
            try res.maskFill_(self.mask.?, @as(T, 0));

            return res;
        }
    };
}

pub fn Affine(comptime batch_size: usize, comptime input_size: usize, comptime output_size: usize, comptime T: type) type {
    const Tensor = tensor.Tensor(&.{ batch_size, input_size }, T, .{});
    const TensorW = tensor.Tensor(&.{ input_size, output_size }, T, .{});
    const TensorB = tensor.Tensor(&.{ 1, output_size }, T, .{});
    const TensorG = tensor.Tensor(&.{ batch_size, output_size }, T, .{});

    return struct {
        w: TensorW,
        b: TensorB,
        x: ?Tensor = null,
        dw: ?TensorW = null,
        db: ?TensorB = null,

        const Self = @This();

        pub fn take_dinfo(self: *Self) struct { dw: ?TensorW, db: ?TensorB } {
            const dw_r = self.dw;
            const dw_b = self.db;

            self.dw = null;
            self.db = null;

            return .{
                .dw = dw_r,
                .db = dw_b,
            };
        }

        pub fn deinit(self: *Self) void {
            self.w.deinit();

            self.b.deinit();

            if (self.x) |x_r| {
                x_r.deinit();
            }

            if (self.dw) |dw_r| {
                dw_r.deinit();
            }

            if (self.db) |db_r| {
                db_r.deinit();
            }
        }

        pub fn init(w: TensorW, b: TensorB) Self {
            return Self{
                .w = w,
                .b = b,
            };
        }

        pub fn forward(self: *Self, x: *const Tensor) !TensorG {
            const x_c = try x.clone();

            if (self.x) |x_r| {
                x_r.deinit();
            }
            self.x = x_c;

            var out = try x.matmul(output_size, &self.w);

            const b_b = self.b.broadcastTo(&.{ batch_size, output_size });
            defer b_b.deinit();

            out.add_(&b_b);

            return out;
        }

        pub fn backward(self: *Self, dout: *const TensorG) !Tensor {
            const w_t = self.w.transpose();
            defer w_t.deinit();

            // std.debug.print("dout: {f} w_t: {f}\n", .{ dout, w_t });
            const dx = try dout.matmul(input_size, &w_t);
            // std.debug.print("dx: {f}\n", .{dx});

            const x_t = self.x.?.transpose();
            defer x_t.deinit();

            const n_dw = try x_t.matmul(output_size, dout);
            const n_db = try dout.sum(0);

            if (self.dw) |dwr| {
                dwr.deinit();
            }
            if (self.db) |dbr| {
                dbr.deinit();
            }

            self.dw = n_dw;
            self.db = n_db;

            return dx;
        }
    };
}

pub fn SoftmaxWithLoss(comptime shape: []const usize, comptime T: type) type {
    const Tensor = tensor.Tensor(shape, T, .{});

    return struct {
        y: ?Tensor = null,
        t: ?Tensor = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            if (self.y) |dw_r| {
                dw_r.deinit();
            }
            if (self.t) |db_r| {
                db_r.deinit();
            }
        }

        pub fn init() Self {
            return Self{};
        }

        pub fn forward(self: *Self, x: *const Tensor, t: *const Tensor) !T {
            if (self.y) |y_r| {
                y_r.deinit();
            }
            if (self.t) |t_r| {
                t_r.deinit();
            }

            // log.print(@src(), "x: {f} t: {f}\n", .{ x.layout, t.layout });
            const loss = try x.crossEntropyLogits(t);
            defer loss.deinit();

            self.y = try x.softmax();
            // std.debug.print("self_y: {f}\n", .{self.y.?});
            self.t = try t.clone();
            // log.print(@src(), "self_t: {f}\n", .{self.t.?});

            // const loss = try self.y.?.crossEntropy(&self.t.?);

            return try loss.dataItem();
        }

        pub fn backward(self: *Self) !Tensor {
            const batch_size = self.t.?.shape()[0];

            // std.debug.print("y: {f} t: {f}\n", .{ self.y.?, self.t.? });
            var dx = try self.y.?.sub(self.t.?);
            dx.divScalar_(@as(T, @floatFromInt(batch_size)));

            return dx;
        }
    };
}

fn TwoDimDemo(comptime batch_size: usize, comptime input_size: usize, comptime hidden_size: usize, comptime output_size: usize) type {
    const Tensor2 = tensor.Tensor(&.{ batch_size, input_size }, f64, .{});
    const AffineI = Affine(batch_size, input_size, hidden_size, f64);
    const Affine2I = Affine(batch_size, hidden_size, output_size, f64);
    const ReluI = Relu(&.{ batch_size, hidden_size }, f64);
    const SwlI = SoftmaxWithLoss(&.{ batch_size, output_size }, f64);

    return struct {
        const TestNet = struct {
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

            fn init(allocator: std.mem.Allocator) !Self {
                var w = try tensor.randNorm(allocator, [2]usize{ input_size, hidden_size }, 0.0, 1.0);
                try w.mul_(0.01);
                var b = try tensor.zeros(allocator, f64, [2]usize{ 1, hidden_size });
                try b.mul_(0.01);

                const w2 = try tensor.randNorm(allocator, [2]usize{ hidden_size, output_size }, 0.0, 1.0);
                const b2 = try tensor.zeros(allocator, f64, [2]usize{ 1, output_size });

                return Self{
                    .allocator = allocator,
                    .relu = ReluI.init(),
                    .affine = AffineI.init(w, b),
                    .affine2 = Affine2I.init(w2, b2),
                    .last_layer = SwlI.init(),
                };
            }

            pub fn forward(self: *Self, x: *const Tensor2) !Tensor2 {
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

            pub fn loss(self: *Self, x: *const Tensor2, t: *const Tensor2) !f64 {
                var f1 = try self.forward(x);
                defer f1.deinit();

                const res = try self.last_layer.forward(&f1, t);
                // const res = try f1.crossEntropyLogits(t);
                defer res.deinit();

                return try res.dataItem();
            }

            pub fn backward_orig(self: *Self, x: *const Tensor2, t: *const Tensor2) !struct { dw: Tensor2, db: Tensor2, dw2: Tensor2, db2: Tensor2 } {
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

            pub fn backward(self: *Self, x: *const Tensor2, t: *const Tensor2) !struct { dw: Tensor2, db: Tensor2 } {
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

            pub fn numericalGradient(self: *Self, x: *const Tensor2, t: *const Tensor2) !struct { dw: Tensor2, db: Tensor2, dw2: Tensor2, db2: Tensor2 } {
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

    const input_dim = 784;
    const hidden_dim = 50;
    const output_dim = 10;
    const batch_size = 50;

    const x = try tensor.randNorm(allocator, &.{ batch_size, input_dim }, 0.0, 1.0);
    defer x.deinit();

    const ta = try tensor.rand(allocator, &.{batch_size}, 0, output_dim);
    defer ta.deinit();
    const t = try ta.oneHot(f64, output_dim);
    std.debug.print("t: {f}\n", .{t});
    // const t = try tensor.fromArray(allocator, [batch_size][output_dim]f64{
    //     .{ 0.0, 0.0, 1.0, 0.0, 0.0 },
    //     .{ 1.0, 0.0, 0.0, 0.0, 0.0 },
    // });
    defer t.deinit();

    var net = try TwoDimDemo.TestNet.init(allocator, input_dim, hidden_dim, output_dim);
    defer net.deinit();

    // const begin = std.time.milliTimestamp();
    // for (0..10000) |_| {
    //     _ = try net.loss(&x, &t);
    // }
    // const end = std.time.milliTimestamp();
    // std.debug.print("Loss computation time: {d} ms\n", .{end - begin});

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

test "affine_relu" {
    const allocator = std.testing.allocator;

    const AffineT = Affine(2, f64);
    const ReluT = Relu(2, f64);
    const SoftWithLossT = SoftmaxWithLoss(2, f64);

    const w = try tensor.arange(allocator, f64, .{ .end = 10 });
    defer w.deinit();

    const w1 = try w.reshape([2]usize{ 2, 5 });
    // defer w1.deinit();

    const ab = try tensor.arange(allocator, f64, .{ .end = 5 });
    defer ab.deinit();

    const ab1 = try ab.reshape([2]usize{ 1, 5 });
    // defer ab1.deinit();

    var affine = AffineT.init(w1, ab1);
    defer affine.deinit();

    var relu = ReluT.init();
    defer relu.deinit();

    var swl = SoftWithLossT.init();
    defer swl.deinit();

    const x = try tensor.arange(allocator, f64, .{ .end = 6 });
    defer x.deinit();

    const x1 = try x.reshape([2]usize{ 3, 2 });
    defer x1.deinit();

    const t = try tensor.fromArray(allocator, [3][5]f64{
        .{ 0, 0, 0, 0, 1 },
        .{ 0, 0, 0, 1, 0 },
        .{ 1, 0, 0, 0, 0 },
    });
    defer t.deinit();

    std.debug.print("input: {f}\n", .{x1});
    const f0 = try affine.forward(&x1);
    defer f0.deinit();

    const f1 = try relu.forward(&f0);
    defer f1.deinit();

    std.debug.print("f1: {f}\n", .{f1});

    const f2 = try swl.forward(&f1, &t);
    defer f2.deinit();

    std.debug.print("f2: {f}\n", .{f2});

    const b2 = try swl.backward();
    defer b2.deinit();
    std.debug.print("b2: {f}\n", .{b2});

    const b1 = try relu.backward(&b2);
    defer b1.deinit();

    std.debug.print("b1: {f}\n", .{b1});

    const b0 = try affine.backward(&b1);
    defer b0.deinit();

    std.debug.print("b0: {f}\n", .{b0});
    // try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, b0.shapes());
}
