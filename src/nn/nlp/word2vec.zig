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
            contexts: *const tensor.Tensor(&.{ batch_size, SizeExpr.static(2), vocab_size }, T),
            target: *const tensor.Tensor(&.{ batch_size, vocab_size }, T),
        ) !T {
            const c0 = try contexts.sliceView(&.{ .All, .{ .Index = SizeExpr.static(0) }, .All });
            defer c0.deinit();
            const c1 = try contexts.sliceView(&.{ .All, .{ .Index = SizeExpr.static(1) }, .All });
            defer c1.deinit();

            // @compileLog("c0: " ++ std.fmt.comptimePrint("{s}\n", .{comptime shape_expr.compLog(@TypeOf(c0).S)}));
            // @compileLog("c1: " ++ std.fmt.comptimePrint("{s}\n", .{comptime shape_expr.compLog(@TypeOf(c1).S)}));
            // std.debug.print("c0: {f} c1: {f}\n", .{ c0, c1 });

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
                .{ .Std = 0.01 },
            );

            const swl = layer.SoftmaxWithLoss(&.{ batch_size, vocab_size }, T).init();

            return Self{
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

    const ids_len = preprocessed.corpus.items.len;

    const batch_size = comptime SizeExpr.sym(.{ .name = "target_len" });
    const words_len_expr = comptime SizeExpr.sym(.{ .name = "window_len" });

    const target_slice = preprocessed.corpus.items[1 .. ids_len - 1];
    try shape_env.bind(&batch_size.Sym, ids_len - 2);

    const word_len = preprocessed.word_to_id.count();
    try shape_env.bind(&words_len_expr.Sym, word_len);

    const target_t = try tensor.fromDataRef(
        usize,
        allocator,
        target_slice,
        &.{batch_size},
        &shape_env,
    );
    defer target_t.deinit();

    var contexts_init_t = try tensor.zeros(
        allocator,
        usize,
        &.{ batch_size, SizeExpr.static(2) },
        &shape_env,
    );
    defer contexts_init_t.deinit();

    for (preprocessed.corpus.items[1 .. ids_len - 1]) |target_id| {
        const left_context = preprocessed.corpus.items[target_id - 1];
        const right_context = preprocessed.corpus.items[target_id + 1];

        try contexts_init_t.setData([_]usize{ target_id - 1, 0 }, left_context);
        try contexts_init_t.setData([_]usize{ target_id - 1, 1 }, right_context);
    }

    const one_hoted_target = try target_t.oneHot(f32, words_len_expr);
    defer one_hoted_target.deinit();
    const one_hoted_contexts = try contexts_init_t.oneHot(f32, words_len_expr);
    defer one_hoted_contexts.deinit();

    std.debug.print("contexts: {f} target= {f}\n", .{ one_hoted_contexts, one_hoted_target });

    // const batch_size = comptime SizeExpr.static(1);
    // const input_size = comptime SizeExpr.static(7);
    const hidden_size = comptime SizeExpr.static(3);

    const SC = SimpleCBOW(batch_size, words_len_expr, hidden_size, f32);
    var sc = try SC.init(allocator, &shape_env);
    defer sc.deinit();

    const loss_v = try sc.loss(&one_hoted_contexts, &one_hoted_target);
    std.debug.print("loss: {}\n", .{loss_v});
}
