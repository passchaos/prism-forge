const std = @import("std");
const shape_expr = @import("../shape_expr.zig");
const tensor = @import("../tensor.zig");
const layer = @import("../nn/layer.zig");
const nlp_classic = @import("classic.zig");
const optim = @import("../nn/optim.zig");
const utils = @import("../utils.zig");
const plot = @import("../plot.zig");

const ShapeEnv = shape_expr.ShapeEnv;
const SizeExpr = shape_expr.SizeExpr;

pub fn SimpleCBOW(
    comptime batch_size: SizeExpr,
    comptime vocab_size: SizeExpr,
    comptime hidden_size: SizeExpr,
    comptime T: type,
) type {
    return struct {
        const Mat_IN = layer.Matmul(
            batch_size,
            &.{vocab_size},
            hidden_size,
            T,
        );
        const Mat_OUT = layer.Matmul(
            batch_size,
            &.{hidden_size},
            vocab_size,
            T,
        );
        const SWL = layer.SoftmaxWithLoss(
            &.{ batch_size, vocab_size },
            T,
        );

        const Self = @This();

        allocator: std.mem.Allocator,
        in_layer0: Mat_IN,
        in_layer1: Mat_IN,
        out_layer: Mat_OUT,
        swl: SWL,

        pub fn gradient(
            self: *Self,
            contexts: *const tensor.Tensor(&.{ batch_size, SizeExpr.static(2), vocab_size }, T),
            target: *const tensor.Tensor(&.{ batch_size, vocab_size }, T),
        ) !optim.WeightGradView(T) {
            const loss_v = try self.loss(contexts, target);
            std.debug.print("loss: {}\n", .{loss_v});

            const dout = try self.swl.backward();
            defer dout.deinit();

            var dout1 = try self.out_layer.backward(&dout);
            defer dout1.deinit();
            dout1.divScalar_(2.0);

            const in0_dx = try self.in_layer1.backward(&dout1);
            defer in0_dx.deinit();
            const in1_dx = try self.in_layer0.backward(&dout1);
            defer in1_dx.deinit();

            var weights = try self.allocator.alloc(tensor.TensorView(T), 3);
            var grads = try self.allocator.alloc(tensor.TensorView(T), 3);

            weights[0] = try self.in_layer0.w.view();
            weights[1] = try self.in_layer1.w.view();
            weights[2] = try self.out_layer.w.view();
            grads[0] = try self.in_layer0.dw.?.view();
            grads[1] = try self.in_layer1.dw.?.view();
            grads[2] = try self.out_layer.dw.?.view();

            return .{
                .allocator = self.allocator,
                .weights = weights,
                .grads = grads,
            };
        }

        pub fn loss(
            self: *Self,
            contexts: *const tensor.Tensor(&.{ batch_size, SizeExpr.static(2), vocab_size }, T),
            target: *const tensor.Tensor(&.{ batch_size, vocab_size }, T),
        ) !T {
            const c0 = try contexts.sliceView(&.{ .All, .{ .Index = SizeExpr.static(0) }, .All });
            defer c0.deinit();
            const c1 = try contexts.sliceView(&.{ .All, .{ .Index = SizeExpr.static(1) }, .All });
            defer c1.deinit();

            var h0 = try self.in_layer0.forward(&c0);
            defer h0.deinit();

            const h1 = try self.in_layer1.forward(&c1);
            defer h1.deinit();

            h0.add_(&h1);
            h0.divScalar_(2.0);

            const out = try self.out_layer.forward(&h0);
            defer out.deinit();

            const res = try self.swl.forward(&out, target);
            return res;
        }

        pub fn init(allocator: std.mem.Allocator, shape_env: *const ShapeEnv) !Self {
            // @compileLog("size: " ++ std.fmt.comptimePrint("{f}\n", .{Mat_IN.I_S_FLAT}));
            const w_in = try tensor.randNorm(
                allocator,
                &.{ vocab_size, hidden_size },
                shape_env,
                @as(f32, 0.0),
                1.0,
            );

            const in_layer0 = try Mat_IN.initImpl(
                shape_env,
                .He,
                try w_in.clone(),
            );

            const in_layer1 = try Mat_IN.initImpl(
                shape_env,
                .He,
                w_in,
            );

            const out_layer = try Mat_OUT.init(
                allocator,
                shape_env,
                .He,
            );

            const swl = layer.SoftmaxWithLoss(&.{ batch_size, vocab_size }, T).init();

            return Self{
                .allocator = allocator,
                .in_layer0 = in_layer0,
                .in_layer1 = in_layer1,
                .out_layer = out_layer,
                .swl = swl,
            };
        }

        pub fn deinit(self: *Self) void {
            self.in_layer0.deinit();
            self.in_layer1.deinit();
            self.out_layer.deinit();
            self.swl.deinit();
        }
    };
}

test "cbow" {
    const allocator = std.testing.allocator;

    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    const text = "You say goodbye and I say hello.";
    var preprocessed = try nlp_classic.preprocess(allocator, text);
    defer preprocessed.deinit();

    const plot_loop = struct {
        fn plot_func(allocator_a: std.mem.Allocator) void {
            plot.beginPlotLoop(allocator_a) catch unreachable;
        }
    }.plot_func;

    _ = try std.Thread.spawn(
        .{},
        plot_loop,
        .{allocator},
    );
    try plot.beginPlotLoop(allocator);

    // const ids_len = preprocessed.corpus.items.len;

    const batch_size = comptime SizeExpr.sym(.{ .name = "target_len" });
    const words_len_expr = comptime SizeExpr.sym(.{ .name = "window_len" });

    // const target_slice = preprocessed.corpus.items[1 .. ids_len - 1];
    try shape_env.bind(&batch_size.Sym, 2);

    const get_contexts_target = struct {
        fn func(
            comptime batch_size_a: SizeExpr,
            comptime words_len_expr_a: SizeExpr,
            allocator_a: std.mem.Allocator,
            shape_env_a: *const ShapeEnv,
            corpus_a: []const usize,
        ) !struct {
            tensor.Tensor(&.{ batch_size_a, SizeExpr.static(2), words_len_expr_a }, f32),
            tensor.Tensor(&.{ batch_size_a, words_len_expr_a }, f32),
        } {
            const count = try batch_size_a.eval(shape_env_a);
            var rand_indexes = try utils.generateRandInRange(usize, allocator_a, count, 1, corpus_a.len - 1);
            defer rand_indexes.deinit(allocator);

            var contexts_res = try tensor.zeros(allocator, usize, &.{ batch_size_a, SizeExpr.static(2) }, shape_env_a);
            defer contexts_res.deinit();
            var target_res = try tensor.zeros(allocator_a, usize, &.{batch_size_a}, shape_env_a);
            defer target_res.deinit();
            for (rand_indexes.items, 0..) |rand_idx, i| {
                try contexts_res.setData([_]usize{ i, 0 }, corpus_a[rand_idx - 1]);
                try target_res.setData([_]usize{i}, corpus_a[rand_idx]);
                try contexts_res.setData([_]usize{ i, 1 }, corpus_a[rand_idx + 1]);
            }

            return .{
                try contexts_res.oneHot(f32, words_len_expr_a),
                try target_res.oneHot(f32, words_len_expr_a),
            };
            // defer rand_idx
            // const rand_idx = try tensor.rand(allocator, &.{batch_size_a}, shape_env, 1, corpus.len - 1);
            // defer rand_idx.deinit();

            // const rand_idxes = rand_idx.dataSliceRaw();
        }
    }.func;

    const word_len = preprocessed.word_to_id.count();
    try shape_env.bind(&words_len_expr.Sym, word_len);

    const hidden_size = comptime SizeExpr.static(3);
    const SC = SimpleCBOW(batch_size, words_len_expr, hidden_size, f32);
    var sc = try SC.init(allocator, &shape_env);
    defer sc.deinit();

    for (0..100) |i| {
        const one_hoted_contexts, const one_hoted_target = try get_contexts_target(
            batch_size,
            words_len_expr,
            allocator,
            &shape_env,
            preprocessed.corpus.items,
        );
        defer {
            one_hoted_contexts.deinit();
            one_hoted_target.deinit();
        }

        const grad = try sc.gradient(&one_hoted_contexts, &one_hoted_target);
        defer grad.deinit();

        var opt = optim.Sgd(f32).init(0.1);
        try opt.update(grad.weights, grad.grads);

        // try shape_env.bind(&batch_size.Sym, preprocessed.corpus.items.len);

        const loss_v = try sc.loss(&one_hoted_contexts, &one_hoted_target);
        try plot.appendData("word2vec", &.{@floatFromInt(i)}, &.{@floatCast(loss_v)});

        // const opt = optim.Adam(f32).init(lr: T, beta1: T, beta2: T, allocator: Allocator)
    }
}
