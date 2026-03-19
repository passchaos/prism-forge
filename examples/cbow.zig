const std = @import("std");
const prism = @import("prism");

const ShapeEnv = prism.shape_expr.ShapeEnv;
const SizeExpr = prism.shape_expr.SizeExpr;
const nlp = prism.nlp;
const plot = prism.plot;
const tensor = prism.tensor;
const utils = prism.utils;
const nn = prism.nn;

fn cbowImpl(allocator: std.mem.Allocator, iters_num: usize) !void {
    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    const text = "You say goodbye and I say hello.";
    var preprocessed = try nlp.classic.preprocess(allocator, text);
    defer preprocessed.deinit();

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
            defer rand_indexes.deinit(allocator_a);

            var contexts_res = try tensor.zeros(allocator_a, usize, &.{ batch_size_a, SizeExpr.static(2) }, shape_env_a);
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

    const hidden_size = comptime SizeExpr.static(5);
    const SC = nlp.word2vec.SimpleCBOW(batch_size, words_len_expr, hidden_size, f32);
    var sc = try SC.init(allocator, &shape_env);
    defer sc.deinit();

    for (0..iters_num) |i| {
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

        const grad = try sc.graident(&one_hoted_contexts, &one_hoted_target);
        defer grad.deinit();

        var opt = nn.optim.Sgd(f32).init(0.1);
        try opt.update(grad.weights, grad.grads);

        // try shape_env.bind(&batch_size.Sym, preprocessed.corpus.items.len);

        const loss_v = try sc.loss(&one_hoted_contexts, &one_hoted_target);
        try plot.appendData("word2vec", &.{@floatFromInt(i)}, &.{@floatCast(loss_v)});

        // const opt = optim.Adam(f32).init(lr: T, beta1: T, beta2: T, allocator: Allocator)
    }

    const word_vecs = sc.in_layer0.w;

    const pw_iter = preprocessed.id_to_word.keys();

    const index_size = comptime SizeExpr.sym(.{ .name = "index" });

    const index_slice = comptime tensor.SliceExpr.index(index_size);

    for (pw_iter) |word_id| {
        try shape_env.bind(&index_size.Sym, word_id);

        const vec = try word_vecs.sliceView(&.{index_slice});
        defer vec.deinit();
        std.debug.print("vec: {f}\n", .{vec});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const t = try std.Thread.spawn(
        .{},
        cbowImpl,
        .{ allocator, 1000 },
    );

    try plot.beginPlotLoop(allocator);

    t.join();
}
