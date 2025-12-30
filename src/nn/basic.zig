const std = @import("std");
const tensor = @import("../tensor.zig");
const log = @import("../log.zig");
const matmul = @import("../matmul.zig");
const mnist = @import("../mnist.zig");
const plot = @import("../plot.zig");
const layer = @import("layer.zig");

const DT = f64;
fn Tensor2(comptime shape: [2]usize) type {
    return tensor.Tensor(&shape, DT, .{});
}

pub fn function2(
    comptime shape: [2]usize,
    input: Tensor2(shape),
    _: void,
) anyerror!DT {
    var input_iter = input.shapeIter();

    var result: DT = 0;

    while (input_iter.next()) |idx| {
        result += std.math.pow(DT, try input.getData(idx), 2);
    }

    return result;
}

pub fn LossArgument(comptime shape: [2]usize) type {
    return struct {
        x: *const Tensor2(shape),
        t: *const Tensor2(shape),
    };
}

pub fn net_loss(
    comptime shape: [2]usize,
    _: *Tensor2(shape),
    net: anytype,
    ctx: LossArgument(shape),
) anyerror!DT {
    const loss = try net.loss(ctx.x, ctx.t);

    return loss;
}

pub fn SimpleNet(comptime shape: [2]usize) type {
    return struct {
        w: Tensor2(shape),

        const Self = @This();

        fn deinit(self: *const Self) void {
            return self.w.deinit();
        }

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
}

// if f return !T, compile will hang or meet unnormal compile error
pub fn numericalGradient(
    allocator: std.mem.Allocator,
    comptime shape: [2]usize,
    net: anytype,
    ctx: anytype,
    f: fn (
        *Tensor2(shape),
        anytype,
        @TypeOf(ctx),
    ) anyerror!DT,
    tval: *Tensor2(shape),
) !Tensor2(shape) {
    const h = 1e-5;

    var grad = try tensor.zerosLike(allocator, tval.*);

    var x_v_iter = tval.shapeIter();
    while (x_v_iter.next()) |idx| {
        const tmp_val = try tval.getData(idx);

        try tval.setData(idx, tmp_val + h);
        const fxh1 = try f(tval, net, ctx);

        // std.debug.print("tval_v: {f}\n", .{tval_v});
        // std.debug.print("idx: {any} fxh1: {}\n", .{ idx, fxh1 });

        try tval.setData(idx, tmp_val - h);
        const fxh2 = try f(tval, net, ctx);
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

pub fn TwoLayerNet(
    comptime batch_size: usize,
    comptime input_size: usize,
    comptime hidden_size: usize,
    comptime output_size: usize,
) type {
    return struct {
        const TensorII = Tensor2([2]usize{ batch_size, input_size });
        const TensorTI = Tensor2([2]usize{ batch_size, output_size });

        const Affine1I = layer.Affine(batch_size, input_size, hidden_size, DT);
        const ReluI = layer.Relu(&.{ batch_size, hidden_size }, DT);
        const Affine2I = layer.Affine(batch_size, hidden_size, output_size, DT);
        const SoftmaxWithLossI = layer.SoftmaxWithLoss(&.{ batch_size, output_size }, DT);
        const Grad = struct {
            dw1: Tensor2([2]usize{ input_size, hidden_size }),
            db1: Tensor2([2]usize{ 1, hidden_size }),
            dw2: Tensor2([2]usize{ hidden_size, output_size }),
            db2: Tensor2([2]usize{ 1, output_size }),
        };

        // params: std.StringHashMap(Tensor2),
        allocator: std.mem.Allocator,

        affine1: Affine1I,
        relu1: ReluI,
        affine2: Affine2I,
        last_layer: SoftmaxWithLossI,

        const Self = @This();

        fn deinit(self: *Self) void {
            // var map_iter = self.params.iterator();
            // while (map_iter.next()) |v| {
            //     v.value_ptr.deinit();
            // }
            // self.params.deinit();

            self.affine1.deinit();
            self.relu1.deinit();
            self.affine2.deinit();
            self.last_layer.deinit();
        }

        fn init(
            allocator: std.mem.Allocator,
            weight_init_std: DT,
        ) !Self {
            // var params_i = std.StringHashMap(Tensor2).init(allocator);

            var w1 = try tensor.randNorm(
                allocator,
                &.{ input_size, hidden_size },
                0.0,
                1.0,
            );
            w1.mulScalar_(weight_init_std);

            const b1 = try tensor.zeros(allocator, DT, &.{ 1, hidden_size });

            var w2 = try tensor.randNorm(
                allocator,
                &.{ hidden_size, output_size },
                0.0,
                1.0,
            );
            w2.mulScalar_(weight_init_std);

            const b2 = try tensor.zeros(allocator, DT, &.{ 1, output_size });

            // try params_i.put("W1", w1);
            // try params_i.put("b1", b1);
            // try params_i.put("W2", w2);
            // try params_i.put("b2", b2);

            // const w1_c = try w1.clone();
            // const b1_c = try b1.clone();
            // const w2_c = try w2.clone();
            // const b2_c = try b2.clone();

            // log.print(@src(), "init w1 sum: {f}", .{try w1_c.sumAll()});

            const affine1 = Affine1I.init(w1, b1);
            const relu1 = ReluI.init();
            const affine2 = Affine2I.init(w2, b2);
            const last_layer = SoftmaxWithLossI.init();

            return Self{
                // .params = params_i,
                .allocator = allocator,
                .affine1 = affine1,
                .relu1 = relu1,
                .affine2 = affine2,
                .last_layer = last_layer,
            };
        }

        fn predict(self: *Self, x: *const TensorII) !TensorTI {
            var x_1 = try self.affine1.forward(x);
            defer x_1.deinit();

            var x_2 = try self.relu1.forward(&x_1);
            defer x_2.deinit();

            const x_3 = try self.affine2.forward(&x_2);

            return x_3;
        }

        fn loss(self: *Self, x: *const TensorII, t: *const TensorTI) !DT {
            const y = try self.predict(x);
            defer y.deinit();

            const loss_t = try self.last_layer.forward(&y, t);

            return loss_t;
        }

        fn accuracy(self: *Self, x: *const TensorII, t: *const TensorTI) !DT {
            const y = try self.predict(x);
            defer y.deinit();

            const y1 = try y.argMax(1);
            defer y1.deinit();
            const t1 = try t.argMax(1);
            defer t1.deinit();

            const eql_t = try y1.eql(t1);
            defer eql_t.deinit();

            log.print(@src(), "eql_t: {f}\n", .{eql_t});
            var eql_sum = try eql_t.sumAll();
            defer eql_sum.deinit();

            const eql_t_d = try eql_sum.div(@as(DT, @floatFromInt(x.shape()[0])));
            defer eql_t_d.deinit();

            return try eql_t_d.dataItem();
        }

        fn numericalGradientM(self: *Self, x: *const TensorII, t: *const TensorTI) !Grad {
            const w1 = try numericalGradient(self.allocator, self, LossArgument{
                .x = x,
                .t = t,
            }, net_loss, &self.affine1.w);

            const w2 = try numericalGradient(self.allocator, self, LossArgument{
                .x = x,
                .t = t,
            }, net_loss, &self.affine2.w);

            const b1 = try numericalGradient(self.allocator, self, LossArgument{
                .x = x,
                .t = t,
            }, net_loss, &self.affine1.b);

            const b2 = try numericalGradient(self.allocator, self, LossArgument{
                .x = x,
                .t = t,
            }, net_loss, &self.affine2.b);

            return Grad{
                .dw1 = w1,
                .db1 = b1,
                .dw2 = w2,
                .db2 = b2,
            };
        }

        fn gradient(self: *Self, x: *const TensorII, t: *const TensorTI) !Grad {
            _ = try self.loss(x, t);

            const dout = try self.last_layer.backward();
            defer dout.deinit();
            // log.print(@src(), "dout layout: {f}\n", .{dout.layout});

            var dout1 = try self.affine2.backward(&dout);
            defer dout1.deinit();
            // log.print(@src(), "dout1 layout: {f}\n", .{dout1.layout});

            var dout2 = try self.relu1.backward(&dout1);
            defer dout2.deinit();
            // log.print(@src(), "dout2 layout: {f}\n", .{dout2.layout});

            const dout3 = try self.affine1.backward(&dout2);
            defer dout3.deinit();
            // log.print(@src(), "dw1: {f} db1: {f}\n", .{ self.affine1.dw.?, self.affine1.db.? });

            return Grad{
                .dw1 = self.affine1.dw.?,
                .db1 = self.affine1.db.?,
                .dw2 = self.affine2.dw.?,
                .db2 = self.affine2.db.?,
            };
        }
    };
}

pub fn twoLayerNetTrain(allocator: std.mem.Allocator, iters_num: usize, comptime batch_size: usize, learning_rate: f64) !void {
    const datas = try mnist.loadDatas(DT, allocator);

    const train_images = datas.train_images;
    defer train_images.deinit();
    const train_labels = datas.train_labels;
    defer train_labels.deinit();
    const test_images = datas.test_images;
    defer test_images.deinit();
    const test_labels = datas.test_labels;
    defer test_labels.deinit();

    log.print(@src(), "train_images: {f} train_labels: {f}\n", .{ train_images.layout, train_labels.layout });

    // var train_loss_list = try std.ArrayList(DT).initCapacity(allocator, 100);
    // var train_accuracy_list = try std.ArrayList(DT).initCapacity(allocator, 100);
    // var test_accuracy_list = try std.ArrayList(DT).initCapacity(allocator, 100);

    // const iter_per_epoch = @max(train)

    const train_size = train_images.shape()[0];

    log.print(@src(), "train_size= {} batch_size= {} iter_num= {} learning_rate= {}\n", .{
        train_size,
        batch_size,
        iters_num,
        learning_rate,
    });

    var net = try TwoLayerNet(batch_size, 784, 50, 10).init(
        allocator,
        0.01,
    );
    defer net.deinit();

    for (0..iters_num) |idx| {
        const batch_mask = try tensor.rand(allocator, &.{batch_size}, @as(usize, 0), train_size);
        defer batch_mask.deinit();

        const batch_indices = batch_mask.dataSliceToArray();
        // const batch_indices = &.{ 0, 1 };

        const x_batch = try train_images.indexSelect(0, batch_size, batch_indices);
        defer x_batch.deinit();
        const t_batch = try train_labels.indexSelect(0, batch_size, batch_indices);
        defer t_batch.deinit();

        const grads1 = try net.gradient(&x_batch, &t_batch);

        // var grads = try net.numericalGradientM(&x_batch, &t_batch);
        // defer grads.deinit();
        // {
        //     var map_key_iter = grads.keyIterator();
        //     while (map_key_iter.next()) |key| {
        //         var v = grads.get(key.*).?;
        //         const v1 = grads1.get(key.*).?;

        //         // std.debug.print("v : {f}\nv1: {f}\n", .{ v, v1 });

        //         var res = try v.sub(v1);
        //         res.abs_();

        //         const mean_diff = try res.meanAll();
        //         defer mean_diff.deinit();

        //         std.debug.print("key: {s} diff: {}\n", .{ key.*, try mean_diff.dataItem() });
        //     }
        // }

        {
            var grad_dw1 = grads1.dw1;
            // defer grad_dw1.deinit();
            grad_dw1.mulScalar_(learning_rate);
            // log.print(@src(), "grad dw1: {f} sum: {f}\n", .{ grad_dw1, try grad_dw1.sumAll() });
            // log.print(@src(), "affine1 dw: sum= {f}\n", .{try net.affine1.dw.?.sumAll()});
            // log.print(@src(), "grad dw1: sum= {f}\n", .{try grad_dw1.sumAll()});
            net.affine1.w.sub_(&grad_dw1);

            // log.print(@src(), "after grad update affine1 dw: sum: {f}\n", .{try net.affine1.dw.?.sumAll()});

            var grad_db1 = grads1.db1;
            // defer grad_db1.deinit();
            grad_db1.mulScalar_(learning_rate);
            net.affine1.b.sub_(&grad_db1);

            var grad_dw2 = grads1.dw2;
            // defer grad_dw2.deinit();
            grad_dw2.mulScalar_(learning_rate);
            net.affine2.w.sub_(&grad_dw2);

            var grad_db2 = grads1.db2;
            // defer grad_db2.deinit();
            grad_db2.mulScalar_(learning_rate);
            net.affine2.b.sub_(&grad_db2);
        }

        // {
        //     var grads_iter = grads1.iterator();
        //     while (grads_iter.next()) |entry| {
        //         const param = entry.key_ptr;

        //         const grad = entry.value_ptr;
        //         defer grad.deinit();

        //         try grad.mul_(learning_rate);

        //         try net.params.getPtr(param.*).?.sub_(grad);
        //     }
        // }

        const loss_idx = try tensor.arange(allocator, @as(usize, 100), .{});
        defer loss_idx.deinit();

        const idx_loss = loss_idx.dataSliceToArray();

        const loss_x = try test_images.indexSelect(0, 100, idx_loss);
        defer loss_x.deinit();
        const loss_t = try test_labels.indexSelect(0, 100, idx_loss);
        defer loss_t.deinit();
        const loss = try net.loss(&loss_x, &loss_t);

        try plot.appendData("idx", &.{@as(f64, @floatFromInt(idx))}, &.{loss});
        log.print(@src(), "idx: {} loss: {}\n", .{ idx, loss });
    }
}

test "simple net" {
    const allocator = std.testing.allocator;

    const weight = try tensor.fromArray(allocator, [_][3]DT{
        .{ 0.47355232, 0.9977393, 0.84668094 },
        .{ 0.85557411, 0.03563661, 0.69422093 },
    });

    var net = SimpleNet([2]usize{ 2, 3 }).init(weight);
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

    const loss = try net.loss(x, t);
    try std.testing.expectApproxEqAbs(0.9280682857864075, loss, 1e-15);
    log.print(@src(), "loss: {}\n", .{loss});

    const result_t = try numericalGradient(
        allocator,
        &net,
        LossArgument{ .x = x, .t = t },
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

test "two_layer_net" {
    const allocator = std.testing.allocator;

    const datas = try mnist.loadDatas(DT, allocator);

    const train_images = datas.train_images;
    defer train_images.deinit();
    const train_labels = datas.train_labels;
    defer train_labels.deinit();
    const test_images = datas.test_images;
    defer test_images.deinit();
    const test_labels = datas.test_labels;
    defer test_labels.deinit();

    var two_layer_net = try TwoLayerNet(3, 784, 100, 10).init(allocator, 0.01);
    defer two_layer_net.deinit();

    {
        const batch_idx = [3]usize{ 0, 1, 2 };
        const batch_train_images = try train_images.indexSelect(0, 3, batch_idx);
        defer batch_train_images.deinit();
        const batch_train_labels = try train_labels.indexSelect(0, 3, batch_idx);
        defer batch_train_labels.deinit();

        const compute_labels = try two_layer_net.predict(&batch_train_images);
        defer compute_labels.deinit();
        log.print(@src(), "compute labels: {f}\n", .{compute_labels.layout});

        const loss = try two_layer_net.loss(&batch_train_images, &batch_train_labels);
        const accuracy = try two_layer_net.accuracy(&batch_train_images, &batch_train_labels);
        log.print(@src(), "loss: {} accuracy: {}\n", .{ loss, accuracy });

        // var gradient = try two_layer_net.numericalGradientM(batch_train_images, batch_train_labels);
        var gradient = try two_layer_net.gradient(&batch_train_images, &batch_train_labels);
        defer gradient.deinit();

        var grad_iter = gradient.iterator();

        while (grad_iter.next()) |entry| {
            defer entry.value_ptr.deinit();
            log.print(@src(), "grad: key= {s} value= {f}\n", .{ entry.key_ptr.*, entry.value_ptr.layout });
        }
    }
}
