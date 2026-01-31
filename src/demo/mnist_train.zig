const std = @import("std");
const tensor = @import("../tensor.zig");
const shape_expr = @import("../shape_expr.zig");
const optim = @import("../nn/optim.zig");
const log = @import("../log.zig");
const plot = @import("../plot.zig");

const conv_net = @import("conv_net.zig");
const mnist = @import("../mnist.zig");

const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;

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

    {
        const C = comptime SizeExpr.static(1);
        const H = comptime SizeExpr.static(28);
        const W = comptime SizeExpr.static(28);
        const FN = comptime SizeExpr.static(30);
        const FP = comptime [2]SizeExpr{ SizeExpr.static(5), SizeExpr.static(5) };
        const PADS = comptime [4]SizeExpr{
            SizeExpr.static(0),
            SizeExpr.static(0),
            SizeExpr.static(0),
            SizeExpr.static(0),
        };
        const STRIDE = comptime SizeExpr.static(1);

        const train_images_conv = try train_images.reshape(&.{ train_data_count_expr, C, H, W });
        defer train_images_conv.deinit();
        const test_images_conv = try test_images.reshape(&.{ test_data_count_expr, C, H, W });
        defer test_images_conv.deinit();

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
            const batch_mask = try tensor.rand(
                allocator,
                &.{batch_size_expr},
                &shape_env,
                @as(usize, 0),
                train_size,
            );
            defer batch_mask.deinit();

            const batch_indices = batch_mask.dataSliceRaw();

            const x_batch = try train_images_conv.indexSelect(0, batch_size_expr, batch_indices);
            defer x_batch.deinit();
            const t_batch = try train_labels.indexSelect(0, batch_size_expr, batch_indices);
            defer t_batch.deinit();

            var adam = optimizer_t{ .ADAM = optim.Adam(DT).init(learning_rate / 10.0, 0.9, 0.999, allocator) };
            defer adam.deinit();

            const weight_grad = try conv_net1.gradient(&x_batch, &t_batch);
            defer weight_grad.deinit();
            try adam.update(weight_grad.weights, weight_grad.grads);

            {
                const loss_idx = try tensor.arange(allocator, @as(usize, batch_size), shape_expr.makeSymbol(.{ .name = "len" }), &shape_env, .{});
                defer loss_idx.deinit();

                const idx_loss = loss_idx.dataSliceRaw();
                const loss_x = try test_images_conv.indexSelect(0, batch_size_expr, idx_loss);
                defer loss_x.deinit();
                const loss_t = try test_labels.indexSelect(0, batch_size_expr, idx_loss);
                defer loss_t.deinit();

                const loss = try conv_net1.loss(&loss_x, &loss_t);
                try plot.appendData("ConvNet", &.{@as(f64, @floatFromInt(idx))}, &.{loss});

                log.print(@src(), "ConvNet: idx= {} loss= {}\n", .{ idx, loss });
            }
        }
    }
}

test "train net" {
    const allocator = std.testing.allocator;

    try trainNet(allocator, 1, 10, 0.01);
}
