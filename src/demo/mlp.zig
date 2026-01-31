const std = @import("std");
const tensor = @import("../tensor.zig");
const shape_expr = @import("../shape_expr.zig");
const layer = @import("../nn/layer.zig");
const optim = @import("../nn/optim.zig");

const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;

pub fn MultiLayerNet(
    comptime batch_size: SizeExpr,
    comptime input_size: []const SizeExpr,
    comptime hidden_sizes: []const SizeExpr,
    comptime output_size: SizeExpr,
    comptime T: type,
) type {
    return struct {
        const InputT = tensor.Tensor(&[1]SizeExpr{batch_size} ++ input_size, T);
        const LabelT = tensor.Tensor(&.{ batch_size, output_size }, T);

        const OutputLayer = layer.Affine(batch_size, &.{hidden_sizes[hidden_sizes.len - 1]}, output_size, T);
        const SoftmaxWithLossLayer = layer.SoftmaxWithLoss(&.{ batch_size, output_size }, T);

        allocator: std.mem.Allocator,

        hidden_affine_layers: []*anyopaque,
        hidden_relu_layers: []*anyopaque,

        use_dropout: bool,
        hidden_dropout_layers: []*anyopaque,

        output_layer: OutputLayer,
        softmax_with_loss: SoftmaxWithLossLayer,

        const Self = @This();

        pub fn gradient(self: *Self, x: *const InputT, t: *const LabelT) !optim.WeightGradView(T) {
            _ = try self.loss(x, t);

            const dout = try self.softmax_with_loss.backward();
            defer dout.deinit();

            // log.print(@src(), "dout layout: {f}\n", .{dout.layout});

            var dout1 = try self.output_layer.backward(&dout);

            // log.print(@src(), "dout1 layout: {f}\n", .{dout1.layout});

            var tmp_grad: *const anyopaque = &dout1;

            inline for (0..hidden_sizes.len) |i| {
                const reverse_idx = hidden_sizes.len - 1 - i;

                const input_size_i = if (i == hidden_sizes.len - 1) input_size else &.{hidden_sizes[reverse_idx - 1]};
                const output_size_i = hidden_sizes[reverse_idx];

                const Affine = layer.Affine(batch_size, input_size_i, output_size_i, T);
                const Relu = layer.Relu(&.{ batch_size, output_size_i }, T);
                const Dropout = layer.Dropout(&.{ batch_size, output_size_i }, T);

                const grad: *const tensor.Tensor(&.{ batch_size, output_size_i }, T) = @ptrCast(@alignCast(tmp_grad));
                defer grad.deinit();

                const relu: *Relu = @ptrCast(@alignCast(self.hidden_relu_layers[reverse_idx]));
                const affine: *Affine = @ptrCast(@alignCast(self.hidden_affine_layers[reverse_idx]));

                if (self.use_dropout) {
                    const dropout: *Dropout = @ptrCast(@alignCast(self.hidden_dropout_layers[reverse_idx]));

                    const grad1 = try dropout.backward(grad);
                    defer grad1.deinit();
                    const grad2 = try relu.backward(&grad1);
                    defer grad2.deinit();
                    const grad3 = try affine.backward(&grad2);

                    if (i == hidden_sizes.len - 1) {
                        grad3.deinit();
                    } else {
                        tmp_grad = &grad3;
                    }
                } else {
                    const grad1 = try relu.backward(grad);
                    defer grad1.deinit();
                    const grad2 = try affine.backward(&grad1);

                    // release last tmp_grad variable
                    if (i == hidden_sizes.len - 1) {
                        grad2.deinit();
                    } else {
                        tmp_grad = &grad2;
                    }
                }
            }

            return self.weightGradView();
        }

        pub fn loss(self: *Self, x: *const InputT, t: *const LabelT) !T {
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

        fn weightGradView(self: *Self) !optim.WeightGradView(T) {
            const count = hidden_sizes.len + 1;

            var weights = try self.allocator.alloc(tensor.TensorView(T), count * 2);
            var grads = try self.allocator.alloc(tensor.TensorView(T), count * 2);

            comptime var tmp_size = input_size;
            inline for (self.hidden_affine_layers, hidden_sizes, 0..) |affine, hidden_size, i| {
                const Affine = layer.Affine(batch_size, tmp_size, hidden_size, T);

                var affine_layer: *Affine = @ptrCast(@alignCast(affine));

                weights[i * 2] = try affine_layer.w.view();
                weights[i * 2 + 1] = try affine_layer.b.view();
                grads[i * 2] = try affine_layer.dw.?.view();
                grads[i * 2 + 1] = try affine_layer.db.?.view();

                tmp_size = &.{hidden_size};
            }

            weights[(count - 1) * 2] = try self.output_layer.w.view();
            weights[(count - 1) * 2 + 1] = try self.output_layer.b.view();
            grads[(count - 1) * 2] = try self.output_layer.dw.?.view();
            grads[(count - 1) * 2 + 1] = try self.output_layer.db.?.view();

            return .{ .allocator = self.allocator, .weights = weights, .grads = grads };
        }

        pub fn deinit(self: *Self) void {
            comptime var tmp_size = input_size;
            inline for (hidden_sizes, self.hidden_affine_layers, self.hidden_relu_layers, self.hidden_dropout_layers) |hidden_size, affine, relu, dropout| {
                const Affine = layer.Affine(batch_size, tmp_size, hidden_size, T);
                const Relu = layer.Relu(&.{ batch_size, hidden_size }, T);
                const Dropout = layer.Dropout(&.{ batch_size, hidden_size }, T);

                var affine_layer: *Affine = @ptrCast(@alignCast(affine));
                var relu_layer: *Relu = @ptrCast(@alignCast(relu));

                affine_layer.deinit();
                relu_layer.deinit();

                if (self.use_dropout) {
                    var dropout_layer: *Dropout = @ptrCast(@alignCast(dropout));
                    dropout_layer.deinit();

                    self.allocator.destroy(dropout_layer);
                }

                self.allocator.destroy(affine_layer);
                self.allocator.destroy(relu_layer);

                tmp_size = &.{hidden_size};
            }
            self.allocator.free(self.hidden_affine_layers);
            self.allocator.free(self.hidden_relu_layers);
            self.allocator.free(self.hidden_dropout_layers);

            self.output_layer.deinit();
            self.softmax_with_loss.deinit();
        }

        pub fn init(
            allocator: std.mem.Allocator,
            weight_init_std: layer.AffineWeight(T),
            shape_env: *const ShapeEnv,
            dropout_ratio: ?f32,
        ) !Self {
            const hidden_affine_layers = try allocator.alloc(*anyopaque, hidden_sizes.len);
            const hidden_relu_layers = try allocator.alloc(*anyopaque, hidden_sizes.len);
            const hidden_dropout_layers = try allocator.alloc(*anyopaque, hidden_sizes.len);

            comptime var tmp_size = input_size;

            inline for (hidden_sizes, 0..) |hidden_size, i| {
                const Affine = layer.Affine(batch_size, tmp_size, hidden_size, T);
                const affine = try allocator.create(Affine);

                affine.* = try Affine.init(allocator, shape_env, weight_init_std);

                const Relu = layer.Relu(&.{ batch_size, hidden_size }, T);
                const relu = try allocator.create(Relu);
                relu.* = Relu.init();

                if (dropout_ratio) |ratio| {
                    const Dropout = layer.Dropout(&.{ batch_size, hidden_size }, T);
                    const dropout = try allocator.create(Dropout);
                    dropout.* = Dropout.init(ratio);
                    hidden_dropout_layers[i] = dropout;
                }

                hidden_affine_layers[i] = affine;
                hidden_relu_layers[i] = relu;

                tmp_size = &.{hidden_size};
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

        fn predict(self: *Self, x: *const InputT) !LabelT {
            var tmp_val: *const anyopaque = x;

            comptime var tmp_size = input_size;
            inline for (hidden_sizes, self.hidden_affine_layers, self.hidden_relu_layers, self.hidden_dropout_layers) |hidden_size, affine, relu, dropout| {
                const Affine = layer.Affine(batch_size, tmp_size, hidden_size, T);
                const Relu = layer.Relu(&.{ batch_size, hidden_size }, T);
                const Dropout = layer.Dropout(&.{ batch_size, hidden_size }, T);

                var affine_layer: *Affine = @ptrCast(@alignCast(affine));
                var relu_layer: *Relu = @ptrCast(@alignCast(relu));

                const input: *const tensor.Tensor(&[1]SizeExpr{batch_size} ++ tmp_size, T) = @ptrCast(@alignCast(tmp_val));

                // log.print(@src(), "begin affine predict: layout= {f} input= {f}\n", .{ affine_layer.w.layout, input.layout });
                const affine_output = try affine_layer.forward(input);
                defer affine_output.deinit();

                // log.print(@src(), "begin relu predict\n", .{});
                const relu_output = try relu_layer.forward(&affine_output);

                if (self.use_dropout) {
                    var dropout_layer: *Dropout = @ptrCast(@alignCast(dropout));
                    const dropout_output = try dropout_layer.forward(&relu_output);

                    relu_output.deinit();
                    tmp_val = &dropout_output;
                } else {
                    tmp_val = &relu_output;
                }

                tmp_size = &.{hidden_size};
            }

            const hidden_output: *const tensor.Tensor(&[1]SizeExpr{batch_size} ++ tmp_size, T) = @ptrCast(@alignCast(tmp_val));
            defer hidden_output.deinit();

            const final_output = try self.output_layer.forward(hidden_output);
            return final_output;
        }
    };
}

test "two_layer_net" {
    const mnist = @import("../mnist.zig");
    const log = @import("../log.zig");

    const allocator = std.testing.allocator;

    const train_data_count_expr = comptime SizeExpr.sym(.{ .name = "train_data_count" });
    const test_data_count_expr = comptime SizeExpr.sym(.{ .name = "test_data_count" });
    const image_data_len_expr = comptime SizeExpr.sym(.{ .name = "image_data_len" });
    const num_classes_expr = comptime SizeExpr.sym(.{ .name = "num_classes" });

    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    try shape_env.bindGlobal(&num_classes_expr.Sym, 10);

    const datas = try mnist.loadDatas(
        f32,
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

    var two_layer_net = try MultiLayerNet(
        SizeExpr.static(3),
        &.{image_data_len_expr},
        &.{SizeExpr.static(100)},
        num_classes_expr,
        f32,
    ).init(allocator, .He, &shape_env, null);
    defer two_layer_net.deinit();

    for (0..3) |_| {
        const batch_idx = [3]usize{ 0, 1, 2 };
        const batch_train_images = try train_images.indexSelect(0, SizeExpr.static(3), &batch_idx);
        defer batch_train_images.deinit();
        const batch_train_labels = try train_labels.indexSelect(0, SizeExpr.static(3), &batch_idx);
        defer batch_train_labels.deinit();

        // comptime {
        //     @compileLog("batch_train_images: " ++ std.fmt.comptimePrint(
        //         "shape= {s}\n",
        //         .{shape_expr.compLog(@TypeOf(batch_train_images).S)},
        //     ));
        // }

        const compute_labels = try two_layer_net.predict(&batch_train_images);
        defer compute_labels.deinit();
        log.print(@src(), "compute labels: {f}\n", .{compute_labels.layout});

        const loss = try two_layer_net.loss(&batch_train_images, &batch_train_labels);
        // const accuracy = try two_layer_net.accuracy(&batch_train_images, &batch_train_labels);
        log.print(@src(), "loss: {}\n", .{loss});

        // var gradient = try two_layer_net.numericalGradientM(batch_train_images, batch_train_labels);
        const weight_gradient = try two_layer_net.gradient(&batch_train_images, &batch_train_labels);
        defer weight_gradient.deinit();

        var sgd = optim.Sgd(f32).init(0.1);
        try sgd.update(weight_gradient.weights, weight_gradient.grads);
        // defer gradient.deinit();

        // var grad_iter = gradient.iterator();

        // while (grad_iter.next()) |entry| {
        //     defer entry.value_ptr.deinit();
        //     log.print(@src(), "grad: key= {s} value= {f}\n", .{ entry.key_ptr.*, entry.value_ptr.layout });
        // }
    }
}
