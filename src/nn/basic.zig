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

pub fn MultiLayerNet(
    comptime batch_size: SizeExpr,
    comptime input_size: SizeExpr,
    comptime hidden_sizes: []const SizeExpr,
    comptime output_size: SizeExpr,
) type {
    return struct {
        const TensorII = Tensor2([2]SizeExpr{ batch_size, input_size });
        const TensorTI = Tensor2([2]SizeExpr{ batch_size, output_size });

        const OutputLayer = layer.Affine(batch_size, hidden_sizes[hidden_sizes.len - 1], output_size, DT);
        const SoftmaxWithLossLayer = layer.SoftmaxWithLoss(&.{ batch_size, output_size }, DT);

        allocator: std.mem.Allocator,

        hidden_affine_layers: []*anyopaque,
        hidden_relu_layers: []*anyopaque,

        use_dropout: bool,
        hidden_dropout_layers: []*anyopaque,

        output_layer: OutputLayer,
        softmax_with_loss: SoftmaxWithLossLayer,

        const Self = @This();

        fn deinit(self: *Self) void {
            // var map_iter = self.params.iterator();
            // while (map_iter.next()) |v| {
            //     v.value_ptr.deinit();
            // }
            // self.params.deinit();

            // self.affine1.deinit();
            // self.relu1.deinit();
            // self.affine2.deinit();
            self.softmax_with_loss.deinit();
        }

        fn init(
            allocator: std.mem.Allocator,
            weight_init_std: layer.AffineWeight(DT),
            shape_env: *const ShapeEnv,
            dropout_ratio: ?f32,
        ) !Self {
            const hidden_affine_layers = try allocator.alloc(*anyopaque, hidden_sizes.len);
            const hidden_relu_layers = try allocator.alloc(*anyopaque, hidden_sizes.len);
            const hidden_dropout_layers = try allocator.alloc(*anyopaque, hidden_sizes.len);

            comptime var tmp_size = input_size;

            inline for (hidden_sizes, 0..) |hidden_size, i| {
                const Affine = layer.Affine(batch_size, tmp_size, hidden_size, DT);
                const affine = try allocator.create(Affine);

                affine.* = try Affine.init(allocator, shape_env, weight_init_std);

                const Relu = layer.Relu(&.{ batch_size, hidden_size }, DT);
                const relu = try allocator.create(Relu);
                relu.* = Relu.init();

                if (dropout_ratio) |ratio| {
                    const Dropout = layer.Dropout(&.{ batch_size, hidden_size }, DT);
                    const dropout = try allocator.create(Dropout);
                    dropout.* = Dropout.init(ratio);
                    hidden_dropout_layers[i] = dropout;
                }

                hidden_affine_layers[i] = affine;
                hidden_relu_layers[i] = relu;

                tmp_size = hidden_size;
            }

            const output_layer = try OutputLayer.init(allocator, shape_env, weight_init_std);
            const softmax_with_loss = SoftmaxWithLossLayer.init();

            return Self{
                .allocator = allocator,
                .hidden_affine_layers = hidden_affine_layers,
                .hidden_relu_layers = hidden_relu_layers,
                .use_dropout = if (dropout_ratio) |_| true else false,
                .hidden_dropout_layers = hidden_dropout_layers,
                .output_layer = output_layer,
                .softmax_with_loss = softmax_with_loss,
            };
        }

        fn predict(self: *Self, x: *const TensorII) !TensorTI {
            var tmp_val: *const anyopaque = x;

            comptime var tmp_size = input_size;
            inline for (hidden_sizes, self.hidden_affine_layers, self.hidden_relu_layers, self.hidden_dropout_layers) |hidden_size, affine, relu, dropout| {
                const Affine = layer.Affine(batch_size, tmp_size, hidden_size, DT);
                const Relu = layer.Relu(&.{ batch_size, hidden_size }, DT);
                const Dropout = layer.Dropout(&.{ batch_size, hidden_size }, DT);

                var affine_layer: *Affine = @ptrCast(@alignCast(affine));
                var relu_layer: *Relu = @ptrCast(@alignCast(relu));

                const input: *const tensor.Tensor(&.{ batch_size, tmp_size }, DT) = @ptrCast(@alignCast(tmp_val));

                // log.print(@src(), "begin affine predict: layout= {f} input= {f}\n", .{ affine_layer.w.layout, input.layout });
                const affine_output = try affine_layer.forward(input);

                // log.print(@src(), "begin relu predict\n", .{});
                const relu_output = try relu_layer.forward(&affine_output);

                if (self.use_dropout) {
                    var dropout_layer: *Dropout = @ptrCast(@alignCast(dropout));
                    const dropout_output = try dropout_layer.forward(&relu_output);
                    tmp_val = &dropout_output;
                } else {
                    tmp_val = &relu_output;
                }

                tmp_size = hidden_size;
            }

            const hidden_output: *const tensor.Tensor(&.{ batch_size, tmp_size }, DT) = @ptrCast(@alignCast(tmp_val));

            const final_output = try self.output_layer.forward(hidden_output);
            return final_output;
        }

        fn loss(self: *Self, x: *const TensorII, t: *const TensorTI) !DT {
            const y = try self.predict(x);
            defer y.deinit();

            const loss_t = try self.softmax_with_loss.forward(&y, t);

            return loss_t;
        }

        // fn accuracy(self: *Self, x: *const TensorII, t: *const TensorTI) !DT {
        //     const y = try self.predict(x);
        //     defer y.deinit();

        //     const y1 = try y.argMax(1);
        //     defer y1.deinit();
        //     const t1 = try t.argMax(1);
        //     defer t1.deinit();

        //     const eql_t = try y1.eql(&t1);
        //     defer eql_t.deinit();

        //     log.print(@src(), "eql_t: {f}\n", .{eql_t});
        //     var eql_sum = try eql_t.sumAll();
        //     defer eql_sum.deinit();

        //     var eql_sum_div = try eql_sum.divScalar(@as(DT, @floatFromInt(x.shape()[0])));
        //     defer eql_sum_div.deinit();

        //     return try eql_sum_div.dataItem();
        // }

        // fn numericalGradientM(self: *Self, x: *const TensorII, t: *const TensorTI) !Grad {
        //     const w1 = try numericalGradient(self.allocator, self, LossArgument{
        //         .x = x,
        //         .t = t,
        //     }, net_loss, &self.affine1.w);

        //     const w2 = try numericalGradient(self.allocator, self, LossArgument{
        //         .x = x,
        //         .t = t,
        //     }, net_loss, &self.affine2.w);

        //     const b1 = try numericalGradient(self.allocator, self, LossArgument{
        //         .x = x,
        //         .t = t,
        //     }, net_loss, &self.affine1.b);

        //     const b2 = try numericalGradient(self.allocator, self, LossArgument{
        //         .x = x,
        //         .t = t,
        //     }, net_loss, &self.affine2.b);

        //     return Grad{
        //         .dw1 = w1,
        //         .db1 = b1,
        //         .dw2 = w2,
        //         .db2 = b2,
        //     };
        // }

        fn weightGradView(self: *Self) !struct { []tensor.TensorView(DT), []tensor.TensorView(DT) } {
            const count = hidden_sizes.len + 1;

            var weights = try self.allocator.alloc(tensor.TensorView(DT), count * 2);
            var grads = try self.allocator.alloc(tensor.TensorView(DT), count * 2);

            comptime var tmp_size = input_size;
            inline for (self.hidden_affine_layers, hidden_sizes, 0..) |affine, hidden_size, i| {
                const Affine = layer.Affine(batch_size, tmp_size, hidden_size, DT);

                var affine_layer: *Affine = @ptrCast(@alignCast(affine));

                weights[i * 2] = try affine_layer.w.view();
                weights[i * 2 + 1] = try affine_layer.b.view();
                grads[i * 2] = try affine_layer.dw.?.view();
                grads[i * 2 + 1] = try affine_layer.db.?.view();

                tmp_size = hidden_size;
            }

            weights[(count - 1) * 2] = try self.output_layer.w.view();
            weights[(count - 1) * 2 + 1] = try self.output_layer.b.view();
            grads[(count - 1) * 2] = try self.output_layer.dw.?.view();
            grads[(count - 1) * 2 + 1] = try self.output_layer.db.?.view();

            return .{ weights, grads };
        }

        fn gradient(self: *Self, x: *const TensorII, t: *const TensorTI) !struct { []tensor.TensorView(DT), []tensor.TensorView(DT) } {
            _ = try self.loss(x, t);

            const dout = try self.softmax_with_loss.backward();
            defer dout.deinit();

            // log.print(@src(), "dout layout: {f}\n", .{dout.layout});

            var dout1 = try self.output_layer.backward(&dout);
            defer dout1.deinit();

            // log.print(@src(), "dout1 layout: {f}\n", .{dout1.layout});

            var tmp_grad: *const anyopaque = &dout1;

            inline for (0..hidden_sizes.len) |i| {
                const reverse_idx = hidden_sizes.len - 1 - i;

                const input_size_i = if (i == hidden_sizes.len - 1) input_size else hidden_sizes[reverse_idx - 1];
                const output_size_i = hidden_sizes[reverse_idx];

                const Affine = layer.Affine(batch_size, input_size_i, output_size_i, DT);
                const Relu = layer.Relu(&.{ batch_size, output_size_i }, DT);
                const Dropout = layer.Dropout(&.{ batch_size, output_size_i }, DT);

                const grad: *const tensor.Tensor(&.{ batch_size, output_size_i }, DT) = @ptrCast(@alignCast(tmp_grad));

                const relu: *Relu = @ptrCast(@alignCast(self.hidden_relu_layers[reverse_idx]));
                const affine: *Affine = @ptrCast(@alignCast(self.hidden_affine_layers[reverse_idx]));

                if (self.use_dropout) {
                    const dropout: *Dropout = @ptrCast(@alignCast(self.hidden_dropout_layers[reverse_idx]));

                    const grad1 = try dropout.backward(grad);
                    const grad2 = try relu.backward(&grad1);
                    const grad3 = try affine.backward(&grad2);

                    tmp_grad = &grad3;
                } else {
                    const grad1 = try relu.backward(grad);
                    const grad2 = try affine.backward(&grad1);

                    tmp_grad = &grad2;
                }
            }

            return self.weightGradView();
        }
    };
}

pub fn twoLayerNetTrain(allocator: std.mem.Allocator, iters_num: usize, batch_size: usize, learning_rate: f64) !void {
    const train_data_count_expr = comptime shape_expr.SizeExpr.sym(.{ .name = "train_data_count" });
    const test_data_count_expr = comptime shape_expr.SizeExpr.sym(.{ .name = "test_data_count" });
    const image_data_len_expr = comptime shape_expr.SizeExpr.sym(.{ .name = "image_data_len" });
    const num_classes_expr = comptime shape_expr.SizeExpr.sym(.{ .name = "num_classes" });

    const batch_size_expr = comptime SizeExpr.sym(.{ .name = "batch_size" });

    var shape_env = try ShapeEnv.init(allocator);
    try shape_env.bindGlobal(&batch_size_expr.Sym, batch_size);
    try shape_env.bindGlobal(&num_classes_expr.Sym, 10);

    const datas = try mnist.loadDatas(
        DT,
        allocator,
        train_data_count_expr,
        test_data_count_expr,
        image_data_len_expr,
        num_classes_expr,
        &shape_env,
    );

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

    const optimizer_t = optim.Optimizer(DT);

    const sgd =
        optimizer_t{ .SGD = optim.Sgd(DT).init(learning_rate) };
    const momentum =
        optimizer_t{ .MOMENTUM = optim.Momentum(DT).init(learning_rate, 0.9, allocator) };
    const ada_grad = optimizer_t{ .ADAGRAD = optim.AdaGrad(DT).init(learning_rate, allocator) };
    const adam = optimizer_t{ .ADAM = optim.Adam(DT).init(learning_rate / 10.0, 0.9, 0.999, allocator) };

    _ = [_]optimizer_t{ adam, ada_grad, momentum, sgd };

    var optimizers = [_]optimizer_t{adam};
    const use_dropout = [4]?f32{ null, 0.1, 0.2, 0.5 };

    for (use_dropout) |dropout_ratio| {
        for (&optimizers) |*optimizer| {
            defer optimizer.reset();

            try shape_env.pushScope();
            defer shape_env.popScope();

            var net = try MultiLayerNet(
                batch_size_expr,
                image_data_len_expr,
                &.{SizeExpr.static(50)},
                num_classes_expr,
            ).init(allocator, .He, &shape_env, dropout_ratio);
            defer net.deinit();

            for (0..iters_num) |idx| {
                try shape_env.pushScope();
                defer shape_env.popScope();
                // try shape_env.bind(&batch_size_expr.Sym, batch_size);

                // const batch_mask = try tensor.arange(
                //     allocator,
                //     batch_size,
                //     shape_expr.makeSymbol(.{ .name = "len" }),
                //     &shape_env,
                //     .{},
                // );
                const batch_mask = try tensor.rand(
                    allocator,
                    &.{batch_size_expr},
                    &shape_env,
                    @as(usize, 0),
                    train_size,
                );
                defer batch_mask.deinit();

                const batch_indices = batch_mask.dataSliceRaw();

                const x_batch = try train_images.indexSelect(0, batch_size_expr, batch_indices);
                defer x_batch.deinit();
                const t_batch = try train_labels.indexSelect(0, batch_size_expr, batch_indices);
                defer t_batch.deinit();

                const grads1 = try net.gradient(&x_batch, &t_batch);
                try optimizer.update(grads1.@"0", grads1.@"1");

                const loss_idx = try tensor.arange(
                    allocator,
                    @as(usize, batch_size),
                    shape_expr.makeSymbol(.{ .name = "len" }),
                    &shape_env,
                    .{},
                );
                defer loss_idx.deinit();

                const idx_loss = loss_idx.dataSliceRaw();

                const loss_x = try test_images.indexSelect(0, batch_size_expr, idx_loss);
                defer loss_x.deinit();
                const loss_t = try test_labels.indexSelect(0, batch_size_expr, idx_loss);
                defer loss_t.deinit();
                const loss = try net.loss(&loss_x, &loss_t);

                const tag_str = try std.fmt.allocPrint(allocator, "dropout: {any} optimizer: {s}", .{ dropout_ratio, @tagName(optimizer.*) });

                // if (idx % 10 == 0) {
                //     std.debug.print("shape env: {f}\n", .{shape_env});
                // }

                try plot.appendData(tag_str, &.{@as(f64, @floatFromInt(idx))}, &.{loss});
                log.print(@src(), "{s}: idx= {} loss= {}\n", .{ tag_str, idx, loss });
            }
        }
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

    var shape_env = ShapeEnv.init(allocator);
    defer shape_env.deinit();

    var two_layer_net = try MultiLayerNet(
        SizeExpr.static(3),
        SizeExpr.static(784),
        SizeExpr.static(100),
        SizeExpr.static(10),
    ).init(allocator, 0.01, &shape_env);
    defer two_layer_net.deinit();

    {
        const batch_idx = [3]usize{ 0, 1, 2 };
        const batch_train_images = try train_images.indexSelect(0, SizeExpr.static(3), &batch_idx);
        defer batch_train_images.deinit();
        const batch_train_labels = try train_labels.indexSelect(0, SizeExpr.static(3), &batch_idx);
        defer batch_train_labels.deinit();

        const compute_labels = try two_layer_net.predict(&batch_train_images);
        defer compute_labels.deinit();
        log.print(@src(), "compute labels: {f}\n", .{compute_labels.layout});

        const loss = try two_layer_net.loss(&batch_train_images, &batch_train_labels);
        const accuracy = try two_layer_net.accuracy(&batch_train_images, &batch_train_labels);
        log.print(@src(), "loss: {} accuracy: {}\n", .{ loss, accuracy });

        // var gradient = try two_layer_net.numericalGradientM(batch_train_images, batch_train_labels);
        _ = try two_layer_net.gradient(&batch_train_images, &batch_train_labels);
        // defer gradient.deinit();

        // var grad_iter = gradient.iterator();

        // while (grad_iter.next()) |entry| {
        //     defer entry.value_ptr.deinit();
        //     log.print(@src(), "grad: key= {s} value= {f}\n", .{ entry.key_ptr.*, entry.value_ptr.layout });
        // }
    }
}
