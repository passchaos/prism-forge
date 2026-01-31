const std = @import("std");
const tensor = @import("../tensor.zig");
const shape_expr = @import("../shape_expr.zig");
const layer = @import("../nn/layer.zig");
const optim = @import("../nn/optim.zig");

const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;

pub fn ConvNet(
    comptime N: SizeExpr,
    comptime C: SizeExpr,
    comptime IP: [2]SizeExpr,
    comptime FN: SizeExpr,
    comptime FP: [2]SizeExpr,
    comptime PADS: [4]SizeExpr,
    comptime STRIDE: SizeExpr,
    comptime HIDDEN_SIZE: SizeExpr,
    comptime OUTPUT_SIZE: SizeExpr,
    comptime T: type,
) type {
    return struct {
        const CONV1_F_OP = layer.dimsPadStrideResult(
            &IP,
            FP,
            PADS,
            STRIDE,
        );

        const CONV1_ODS = [4]SizeExpr{
            N,
            FN,
            CONV1_F_OP[0],
            CONV1_F_OP[1],
        };

        const POOLING1_FP = [2]SizeExpr{
            SizeExpr.static(2),
            SizeExpr.static(2),
        };
        const POOLING1_PADS = [4]SizeExpr{
            SizeExpr.static(0),
            SizeExpr.static(0),
            SizeExpr.static(0),
            SizeExpr.static(0),
        };
        const POOLING1_STRIDE = SizeExpr.static(2);
        const POOLING1_F_OP = layer.dimsPadStrideResult(
            &CONV1_F_OP,
            POOLING1_FP,
            POOLING1_PADS,
            POOLING1_STRIDE,
        );
        const POOLING1_ODS = [4]SizeExpr{
            N,
            FN,
            POOLING1_F_OP[0],
            POOLING1_F_OP[1],
        };

        const Conv1 = layer.Convolution(
            N,
            C,
            IP,
            FN,
            FP,
            PADS,
            STRIDE,
            T,
        );
        const Relu1 = layer.Relu(
            &.{ N, FN, CONV1_F_OP[0], CONV1_F_OP[1] },
            T,
        );
        const Pooling1 = layer.Pooling(
            N,
            FN,
            CONV1_F_OP,
            POOLING1_FP,
            POOLING1_PADS,
            POOLING1_STRIDE,
            T,
        );
        const Affine1 = layer.Affine(
            N,
            &.{ FN, POOLING1_F_OP[0], POOLING1_F_OP[1] },
            HIDDEN_SIZE,
            T,
        );
        const Relu2 = layer.Relu(&.{
            N, HIDDEN_SIZE,
        }, T);
        const Affine2 = layer.Affine(
            N,
            &.{HIDDEN_SIZE},
            OUTPUT_SIZE,
            T,
        );
        const SoftmaxWithLoss = layer.SoftmaxWithLoss(
            &.{ N, OUTPUT_SIZE },
            T,
        );

        conv1: Conv1,
        relu1: Relu1,
        pooling1: Pooling1,
        affine1: Affine1,
        relu2: Relu2,
        affine2: Affine2,
        softmax_with_loss: SoftmaxWithLoss,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn gradient(
            self: *Self,
            x: *const tensor.Tensor(&.{ N, C, IP[0], IP[1] }, T),
            t: *const tensor.Tensor(&.{ N, OUTPUT_SIZE }, T),
        ) !optim.WeightGradView(T) {
            _ = try self.loss(x, t);

            const softmax_with_loss_out = try self.softmax_with_loss.backward();
            defer softmax_with_loss_out.deinit();
            const affine2_out = try self.affine2.backward(&softmax_with_loss_out);
            defer affine2_out.deinit();
            const relu2_out = try self.relu2.backward(&affine2_out);
            defer relu2_out.deinit();
            const affine1_out = try self.affine1.backward(&relu2_out);
            defer affine1_out.deinit();
            const pooling1_out = try self.pooling1.backward(&affine1_out);
            defer pooling1_out.deinit();
            const relu1_out = try self.relu1.backward(&pooling1_out);
            defer relu1_out.deinit();
            const conv1_out = try self.conv1.backward(&relu1_out);
            defer conv1_out.deinit();
            // std.debug.print("conv1 out: {f}\n", .{conv1_out});

            var weights = try self.allocator.alloc(tensor.TensorView(T), 6);
            var grads = try self.allocator.alloc(tensor.TensorView(T), 6);

            weights[0] = try self.conv1.w.view();
            grads[0] = try self.conv1.dw.?.view();
            weights[1] = try self.conv1.b.view();
            grads[1] = try self.conv1.db.?.view();
            weights[2] = try self.affine1.w.view();
            grads[2] = try self.affine1.dw.?.view();
            weights[3] = try self.affine1.b.view();
            grads[3] = try self.affine1.db.?.view();
            weights[4] = try self.affine2.w.view();
            grads[4] = try self.affine2.dw.?.view();
            weights[5] = try self.affine2.b.view();
            grads[5] = try self.affine2.db.?.view();

            return optim.WeightGradView(T){
                .allocator = self.allocator,
                .weights = weights,
                .grads = grads,
            };
        }

        pub fn loss(
            self: *Self,
            x: *const tensor.Tensor(&.{ N, C, IP[0], IP[1] }, T),
            t: *const tensor.Tensor(&.{ N, OUTPUT_SIZE }, T),
        ) !T {
            const affine2_out = try self.predict(x);
            defer affine2_out.deinit();
            return try self.softmax_with_loss.forward(&affine2_out, t);
        }

        fn predict(
            self: *Self,
            x: *const tensor.Tensor(&.{ N, C, IP[0], IP[1] }, T),
        ) !tensor.Tensor(&.{ N, OUTPUT_SIZE }, T) {
            const conv1_out = try self.conv1.forward(x);
            defer conv1_out.deinit();

            // comptime {
            //     @setEvalBranchQuota(30000);
            //     @compileLog("conv1_out shape: " ++ std.fmt.comptimePrint(
            //         "{s} {s}\n",
            //         .{ shape_expr.compLog(@TypeOf(conv1_out).S), shape_expr.compLog(Relu1.S) },
            //     ));
            // }

            const relu1_out = try self.relu1.forward(&conv1_out);
            defer relu1_out.deinit();
            const pooling1_out = try self.pooling1.forward(&relu1_out);
            defer pooling1_out.deinit();
            const affine1_out = try self.affine1.forward(&pooling1_out);
            defer affine1_out.deinit();
            const relu2_out = try self.relu2.forward(&affine1_out);
            defer relu2_out.deinit();
            const affine2_out = try self.affine2.forward(&relu2_out);

            return affine2_out;
        }

        pub fn init(
            allocator: std.mem.Allocator,
            shape_env: *const ShapeEnv,
            weight_init_std: T,
        ) !Self {
            const conv1 = try Conv1.init(
                allocator,
                shape_env,
                weight_init_std,
            );
            const relu1 = Relu1.init();
            const pooling1 = Pooling1.init(
                allocator,
                shape_env,
            );
            const affine1 = try Affine1.init(
                allocator,
                shape_env,
                layer.AffineWeight(T){ .Std = weight_init_std },
            );
            const relu2 = Relu2.init();
            const affine2 = try Affine2.init(
                allocator,
                shape_env,
                layer.AffineWeight(T){ .Std = weight_init_std },
            );
            const softmax_with_loss = SoftmaxWithLoss.init();

            return Self{
                .conv1 = conv1,
                .relu1 = relu1,
                .pooling1 = pooling1,
                .affine1 = affine1,
                .relu2 = relu2,
                .affine2 = affine2,
                .softmax_with_loss = softmax_with_loss,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.conv1.deinit();
            self.relu1.deinit();
            self.pooling1.deinit();
            self.affine1.deinit();
            self.relu2.deinit();
            self.affine2.deinit();
            self.softmax_with_loss.deinit();
        }
    };
}
