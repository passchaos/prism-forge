const std = @import("std");
const prism = @import("prism_forge");

const tensor = prism.tensor;
const optim = prism.optim;
const log = prism.log;
const mnist = prism.mnist;
const mlp = prism.mlp;
const conv_net = prism.conv_net;
const plot = prism.plot;

const SizeExpr = prism.shape_expr.SizeExpr;
const ShapeEnv = prism.shape_expr.ShapeEnv;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    const t1 = try std.Thread.spawn(
        .{},
        trainNet,
        .{ allocator, 10000, 100, 0.001 },
    );
    try prism.plot.beginPlotLoop(allocator);
    t1.join();
    std.debug.print("Hello, world!\n", .{});
}

pub fn trainNet(
    allocator: std.mem.Allocator,
    iters_num: usize,
    batch_size: usize,
    learning_rate: f32,
) !void {
    const train_data_count_expr = comptime SizeExpr.sym(.{ .name = "train_data_count" });
    const test_data_count_expr = comptime SizeExpr.sym(.{ .name = "test_data_count" });
    const image_data_len_expr = comptime SizeExpr.sym(.{ .name = "image_data_len" });
    const num_classes_expr = comptime SizeExpr.sym(.{ .name = "num_classes" });

    const batch_size_expr = comptime SizeExpr.sym(.{ .name = "batch_size" });

    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    try shape_env.bindGlobal(&batch_size_expr.Sym, batch_size);
    try shape_env.bindGlobal(&num_classes_expr.Sym, 10);

    const DT = @TypeOf(learning_rate);

    const datas = try prism.mnist.loadDatas(
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

    const C = comptime SizeExpr.static(1);
    const H = comptime SizeExpr.static(28);
    const W = comptime SizeExpr.static(28);

    const train_images_c = try train_images.reshape(&.{ train_data_count_expr, C, H, W });
    defer train_images_c.deinit();
    const test_images_c = try test_images.reshape(&.{ test_data_count_expr, C, H, W });
    defer test_images_c.deinit();

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

    {
        var net = try mlp.MultiLayerNet(
            batch_size_expr,
            &.{ C, H, W },
            &.{SizeExpr.static(100)},
            num_classes_expr,
            DT,
        ).init(allocator, .He, &shape_env, null);
        defer net.deinit();

        for (0..iters_num) |idx| {
            try shape_env.bindGlobal(&batch_size_expr.Sym, batch_size);

            const batch_mask = try tensor.rand(
                allocator,
                &.{batch_size_expr},
                &shape_env,
                @as(usize, 0),
                train_size,
            );
            defer batch_mask.deinit();

            const batch_indices = batch_mask.dataSliceRaw();

            const x_batch = try train_images_c.indexSelect(0, batch_size_expr, batch_indices);
            defer x_batch.deinit();
            const t_batch = try train_labels.indexSelect(0, batch_size_expr, batch_indices);
            defer t_batch.deinit();

            const epoch_count = @min(idx / (train_size / batch_size) + 1, 5);
            var lr = learning_rate;
            for (0..epoch_count) |_| {
                lr *= 0.5;
            }

            var adam = optimizer_t{ .ADAM = optim.Adam(DT).init(lr, 0.9, 0.999, allocator) };
            defer adam.deinit();

            // log.print(@src(), "mlp, before train\n", .{});
            const grads1 = try net.gradient(&x_batch, &t_batch);
            defer grads1.deinit();
            // log.print(@src(), "before gradient update\n", .{});

            try adam.update(grads1.weights, grads1.grads);

            try shape_env.bindGlobal(&batch_size_expr.Sym, test_images.shape()[0]);
            const loss_x = try test_images_c.reshape(&.{ batch_size_expr, C, H, W });
            defer loss_x.deinit();
            const loss_t = try test_labels.reshape(&.{ batch_size_expr, num_classes_expr });
            defer loss_t.deinit();
            // const loss_idx = try tensor.arange(
            //     allocator,
            //     @as(usize, test_images.shape()[0]),
            //     shape_expr.makeSymbol(.{ .name = "len" }),
            //     &shape_env,
            //     .{},
            // );
            // defer loss_idx.deinit();

            // const idx_loss = loss_idx.dataSliceRaw();

            // const loss_x = try test_images_c.indexSelect(0, batch_size_expr, idx_loss);
            // defer loss_x.deinit();
            // const loss_t = try test_labels.indexSelect(0, batch_size_expr, idx_loss);
            // defer loss_t.deinit();

            // std.debug.print("loss: x= {f} t= {f}\n", .{ loss_x.layout, loss_t.layout });
            // log.print(@src(), "before loss get\n", .{});
            const loss = try net.loss(&loss_x, &loss_t);
            // log.print(@src(), "after loss get\n", .{});
            const accuracy = try net.accuracy(&loss_x, &loss_t);

            try plot.appendData("MLP Loss", &.{@as(f64, @floatFromInt(idx))}, &.{loss});
            try plot.appendData("MLP Accuracy", &.{@as(f64, @floatFromInt(idx))}, &.{accuracy});

            log.print(@src(), "MLP: idx= {} loss= {} accuracy= {}\n", .{ idx, loss, accuracy });
        }
    }

    {
        const FN = comptime SizeExpr.static(30);
        const FP = comptime [2]SizeExpr{ SizeExpr.static(5), SizeExpr.static(5) };
        const PADS = comptime [4]SizeExpr{
            SizeExpr.static(0),
            SizeExpr.static(0),
            SizeExpr.static(0),
            SizeExpr.static(0),
        };
        const STRIDE = comptime SizeExpr.static(1);

        var conv_net1 = try conv_net.ConvNet(
            batch_size_expr,
            C,
            [2]SizeExpr{ H, W },
            FN,
            FP,
            PADS,
            STRIDE,
            SizeExpr.static(100),
            num_classes_expr,
            DT,
        ).init(
            allocator,
            &shape_env,
            0.01,
        );
        defer conv_net1.deinit();

        for (0..iters_num) |idx| {
            try shape_env.bindGlobal(&batch_size_expr.Sym, batch_size);
            const batch_mask = try tensor.rand(
                allocator,
                &.{batch_size_expr},
                &shape_env,
                @as(usize, 0),
                train_size,
            );
            defer batch_mask.deinit();

            const batch_indices = batch_mask.dataSliceRaw();

            const x_batch = try train_images_c.indexSelect(0, batch_size_expr, batch_indices);
            defer x_batch.deinit();
            const t_batch = try train_labels.indexSelect(0, batch_size_expr, batch_indices);
            defer t_batch.deinit();

            const epoch_count = idx / (train_size / batch_size);
            var lr_r = learning_rate;
            for (0..epoch_count) |_| {
                lr_r *= 0.7;
            }

            const lr = @max(lr_r, 1e-5);

            var adam = optimizer_t{ .ADAM = optim.Adam(DT).init(lr, 0.9, 0.999, allocator) };
            defer adam.deinit();

            const weight_grad = try conv_net1.gradient(&x_batch, &t_batch);
            defer weight_grad.deinit();
            try adam.update(weight_grad.weights, weight_grad.grads);

            {
                try shape_env.bindGlobal(&batch_size_expr.Sym, test_images.shape()[0]);
                const loss_idx = try tensor.arange(allocator, @as(usize, test_images.shape()[0]), prism.shape_expr.makeSymbol(.{ .name = "len" }), &shape_env, .{});
                defer loss_idx.deinit();

                const idx_loss = loss_idx.dataSliceRaw();
                const loss_x = try test_images_c.indexSelect(0, batch_size_expr, idx_loss);
                defer loss_x.deinit();
                const loss_t = try test_labels.indexSelect(0, batch_size_expr, idx_loss);
                defer loss_t.deinit();

                const loss = try conv_net1.loss(&loss_x, &loss_t);
                const accuracy = try conv_net1.accuracy(&loss_x, &loss_t);
                try plot.appendData("ConvNet Loss", &.{@as(f64, @floatFromInt(idx))}, &.{loss});
                try plot.appendData("ConvNet Accuracy", &.{@as(f64, @floatFromInt(idx))}, &.{accuracy});

                log.print(@src(), "ConvNet: idx= {} lr= {} loss= {} accuracy= {}\n", .{ idx, lr, loss, accuracy });
            }
        }
    }
}
