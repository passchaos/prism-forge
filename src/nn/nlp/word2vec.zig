const std = @import("std");
const shape_expr = @import("../../shape_expr.zig");
const tensor = @import("../../tensor.zig");
const layer = @import("../layer.zig");
const nlp_classic = @import("./classic.zig");

const ShapeEnv = shape_expr.ShapeEnv;
const SizeExpr = shape_expr.SizeExpr;

fn SimpleCBOW(
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

        in_layer0: Mat_IN,
        in_layer1: Mat_IN,
        out_layer: Mat_OUT,
        swl: SWL,

        pub fn loss(
            self: *Self,
            contexts: *const tensor.Tensor(&.{ batch_size, vocab_size }, T),
            targets: *const tensor.Tensor(&.{ batch_size, vocab_size }, T),
        ) !T {
            // const c0 = try contexts.sliceView(&.{ .All, .Index(0) });
            // const h0 = self.in_layer0.forward(x: *const Tensor(T))
            return 10.0;
        }
    };
}

test "cbow" {
    const allocator = std.testing.allocator;

    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    const c0 = try tensor.fromArray(allocator, [1][7]f32{
        [_]f32{ 1, 0, 0, 0, 0, 0, 0 },
    }, &shape_env);
    defer c0.deinit();

    const c1 = try tensor.fromArray(allocator, [1][7]f32{
        [_]f32{ 0, 0, 1, 0, 0, 0, 0 },
    }, &shape_env);
    defer c1.deinit();

    const b_s = comptime SizeExpr.static(1);
    const i_i_s = comptime SizeExpr.static(7);
    const i_o_s = comptime SizeExpr.static(3);
    const o_i_s = comptime SizeExpr.static(3);
    const o_o_s = comptime SizeExpr.static(7);

    const Mat_IN = layer.Matmul(b_s, &.{i_i_s}, i_o_s, f32);
    const Mat_OUT = layer.Matmul(b_s, &.{o_i_s}, o_o_s, f32);

    const w_in = try tensor.randNorm(
        allocator,
        &.{ i_i_s, i_o_s },
        &shape_env,
        @as(f32, 0.0),
        1.0,
    );

    var in_layer0 = try Mat_IN.initImpl(
        &shape_env,
        .He,
        try w_in.clone(),
    );
    defer in_layer0.deinit();

    var in_layer1 = try Mat_IN.initImpl(
        &shape_env,
        .He,
        w_in,
    );
    defer in_layer1.deinit();

    var out_layer = try Mat_OUT.init(
        allocator,
        &shape_env,
        .{ .Std = 0.01 },
    );
    defer out_layer.deinit();

    var h0 = try in_layer0.forward(&c0);
    defer h0.deinit();
    const h1 = try in_layer1.forward(&c1);
    defer h1.deinit();

    h0.add_(&h1);
    h0.divScalar_(2.0);

    const out = try out_layer.forward(&h0);
    defer out.deinit();
    std.debug.print("out: {f}\n", .{out});

    const text = "You say goodbye and I say hello.";
    var preprocessed = try nlp_classic.preprocess(allocator, text);
    defer preprocessed.deinit();

    const ids_len = preprocessed.corpus.items.len;

    const target_len_expr = comptime SizeExpr.sym(.{ .name = "target_len" });
    const window_len_expr = comptime SizeExpr.sym(.{ .name = "window_len" });

    const target_slice = preprocessed.corpus.items[1 .. ids_len - 1];
    try shape_env.bind(&target_len_expr.Sym, ids_len - 2);

    const word_len = preprocessed.word_to_id.count();
    try shape_env.bind(&window_len_expr.Sym, word_len);

    const target_t = try tensor.fromDataRef(
        usize,
        allocator,
        target_slice,
        &.{target_len_expr},
        &shape_env,
    );
    defer target_t.deinit();

    var contexts_init_t = try tensor.zeros(
        allocator,
        usize,
        &.{ target_len_expr, SizeExpr.static(2) },
        &shape_env,
    );
    defer contexts_init_t.deinit();

    for (preprocessed.corpus.items[1 .. ids_len - 1]) |target_id| {
        const left_context = preprocessed.corpus.items[target_id - 1];
        const right_context = preprocessed.corpus.items[target_id + 1];

        try contexts_init_t.setData([_]usize{ target_id - 1, 0 }, left_context);
        try contexts_init_t.setData([_]usize{ target_id - 1, 1 }, right_context);
    }

    const one_hoted_target = try target_t.oneHot(f32, window_len_expr);
    defer one_hoted_target.deinit();
    const one_hoted_contexts = try contexts_init_t.oneHot(f32, window_len_expr);
    defer one_hoted_contexts.deinit();

    std.debug.print("contexts: {f} target= {f}\n", .{ one_hoted_contexts, one_hoted_target });
}
