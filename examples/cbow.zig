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

    const ptb_file = try std.fs.cwd().openFile("assets/ptb.train.txt", .{});
    const ptb_contents_raw = try ptb_file.readToEndAlloc(allocator, (try ptb_file.stat()).size);
    defer allocator.free(ptb_contents_raw);

    var start: usize = 0;
    for (0..100) |_| {
        start = std.mem.indexOfScalarPos(u8, ptb_contents_raw, start, '\n').? + 1;
    }

    const text = ptb_contents_raw[0..start];

    // const text = "You say goodbye and I say hello.";
    var preprocessed = try nlp.classic.preprocess(allocator, text);
    defer preprocessed.deinit();

    std.debug.print("preprocessed: corpus={} words={}\n", .{ preprocessed.corpus.items.len, preprocessed.word_to_id.count() });

    // const ids_len = preprocessed.corpus.items.len;

    const batch_size = comptime SizeExpr.sym(.{ .name = "target_len" });
    const words_len_expr = comptime SizeExpr.sym(.{ .name = "window_len" });

    // const target_slice = preprocessed.corpus.items[1 .. ids_len - 1];

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

    const hidden_size = comptime SizeExpr.static(100);
    const SC = nlp.word2vec.SimpleCBOW(batch_size, words_len_expr, hidden_size, f32);
    var sc = try SC.init(allocator, &shape_env);
    defer sc.deinit();

    for (0..iters_num) |i| {
        try shape_env.bind(&batch_size.Sym, 20);

        const one_hoted_contexts, const full_targets = try get_contexts_target(
            batch_size,
            words_len_expr,
            allocator,
            &shape_env,
            preprocessed.corpus.items,
        );
        defer {
            one_hoted_contexts.deinit();
            full_targets.deinit();
        }

        const grad = try sc.gradient(&one_hoted_contexts, &full_targets);
        defer grad.deinit();

        var opt = nn.optim.Adam(f32).init(0.01, 0.9, 0.999, allocator);
        try opt.update(grad.weights, grad.grads);

        // try shape_env.bind(&batch_size.Sym, preprocessed.corpus.items.len);

        {
            const corpus = preprocessed.corpus.items;
            try shape_env.bind(&batch_size.Sym, corpus.len);

            var contexts_res = try tensor.zeros(allocator, usize, &.{ batch_size, SizeExpr.static(2) }, &shape_env);
            defer contexts_res.deinit();
            var target_res = try tensor.zeros(allocator, usize, &.{batch_size}, &shape_env);
            defer target_res.deinit();

            for (1..corpus.len - 1) |idx| {
                try contexts_res.setData([_]usize{ idx, 0 }, corpus[idx - 1]);
                try target_res.setData([_]usize{idx}, corpus[idx]);
                try contexts_res.setData([_]usize{ idx, 1 }, corpus[idx + 1]);
            }

            const full_contexts = try contexts_res.oneHot(f32, words_len_expr);
            defer full_contexts.deinit();
            const full_target = try target_res.oneHot(f32, words_len_expr);
            defer full_target.deinit();

            const loss_v = try sc.loss(&full_contexts, &full_target);
            try plot.appendData("word2vec", &.{@floatFromInt(i)}, &.{@floatCast(loss_v)});
        }

        // const opt = optim.Adam(f32).init(lr: T, beta1: T, beta2: T, allocator: Allocator)
    }

    const word_vecs = sc.in_layer0.w;

    const pw_iter = preprocessed.id_to_word.keys();

    const index_size = comptime SizeExpr.sym(.{ .name = "index" });

    const index_slice = comptime tensor.SliceExpr.index(index_size);

    for (pw_iter[0..10]) |word_id| {
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
        .{ allocator, 10000 },
    );

    try plot.beginPlotLoop(allocator);

    t.join();
}
