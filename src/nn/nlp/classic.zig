const std = @import("std");
const shape_expr = @import("../../shape_expr.zig");
const tensor = @import("../../tensor.zig");

const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;

pub fn preprocess(allocator: std.mem.Allocator, text: []const u8) !struct {
    corpus: std.ArrayList(usize),
    word_to_id: std.StringArrayHashMap(usize),
    id_to_word: std.AutoArrayHashMap(usize, []const u8),
} {
    const lower_replaced_text = try std.mem.replaceOwned(u8, allocator, text, ".", " .");
    // defer allocator.free(lower_replaced_text);

    for (lower_replaced_text) |*c| {
        c.* = std.ascii.toLower(c.*);
    }
    // const lower_text = try std.ascii.allocLowerString(allocator, text);

    var split_iterator = std.mem.splitScalar(u8, lower_replaced_text, ' ');

    var word_to_id = std.StringArrayHashMap(usize).init(allocator);
    var id_to_word = std.AutoArrayHashMap(usize, []const u8).init(allocator);
    var corpus = try std.ArrayList(usize).initCapacity(allocator, 10);

    var idx: usize = 0;
    while (split_iterator.next()) |word| {
        if (!word_to_id.contains(word)) {
            try word_to_id.put(word, idx);
            try id_to_word.put(idx, word);

            idx += 1;
        }

        if (word_to_id.get(word)) |w_id| {
            try corpus.append(allocator, w_id);
        }
    }

    return .{
        .corpus = corpus,
        .word_to_id = word_to_id,
        .id_to_word = id_to_word,
    };
}

pub fn create_co_matrix(
    allocator: std.mem.Allocator,
    shape_env: *const ShapeEnv,
    corpus: *const std.ArrayList(usize),
    comptime vocab_size: SizeExpr,
    window_size: usize,
) !tensor.Tensor(&.{ vocab_size, vocab_size }, usize) {
    var co_matrix = try tensor.zeros(
        allocator,
        usize,
        &.{ vocab_size, vocab_size },
        shape_env,
    );

    for (corpus.items, 0..) |w_id, i| {
        const start = if (i >= window_size) i - window_size else 0;
        const end = @min(corpus.items.len, i + window_size + 1);

        for (start..end) |j| {
            if (i == j) continue;

            const col_id = corpus.items[j];

            const idx = [2]usize{ w_id, col_id };

            const orig_v = co_matrix.getData(idx) catch 0;
            try co_matrix.setData(idx, orig_v + 1);
        }
    }

    return co_matrix;
}

pub fn mostSimilar(
    allocator: std.mem.Allocator,
    shape_env: *ShapeEnv,
    query: []const u8,
    comptime vocab_size: SizeExpr,
    co_matrix: *const tensor.Tensor(&.{ vocab_size, vocab_size }, usize),
    word_to_id: *const std.StringArrayHashMap(usize),
    top: usize,
) !void {
    const query_id = if (word_to_id.get(query)) |query_id| query_id else return;

    const query_se = comptime SizeExpr.sym(.{ .name = "query" });
    try shape_env.bind(&query_se.Sym, query_id);
    const query_co = try co_matrix.sliceView(&.{tensor.SliceExpr.index(query_se)});

    const WordSimilarity = struct {
        word: []const u8,
        similar: f64,

        fn asc(_: void, a: @This(), b: @This()) bool {
            return a.similar > b.similar;
        }
    };

    var word_similars = try std.ArrayList(WordSimilarity)
        .initCapacity(allocator, 10);
    defer word_similars.deinit(allocator);

    var word_iter = word_to_id.iterator();
    while (word_iter.next()) |entry| {
        const word = entry.key_ptr.*;
        const word_id = entry.value_ptr.*;

        if (std.mem.eql(u8, query, word)) continue;

        const word_se = comptime SizeExpr.sym(.{ .name = "word_inner" });
        try shape_env.bind(&word_se.Sym, word_id);
        const word_co = try co_matrix.sliceView(&.{tensor.SliceExpr.index(word_se)});

        const sim = try query_co.cosineSimilarity(&word_co);
        try word_similars.append(allocator, .{ .word = word, .similar = sim });
    }

    std.mem.sort(WordSimilarity, word_similars.items, void{}, WordSimilarity.asc);

    const top_i = @min(top, word_similars.items.len);
    std.debug.print("[query] {s}\n", .{query});
    for (word_similars.items[0..top_i]) |sim| {
        std.debug.print(" {s}: {}\n", .{ sim.word, sim.similar });
    }
}

pub fn ppmi(
    comptime vocab_size: SizeExpr,
    co_matrix: *const tensor.Tensor(&.{ vocab_size, vocab_size }, usize),
) !tensor.Tensor(&.{ vocab_size, vocab_size }, f64) {
    var co_matrix_f64 = try co_matrix.to(f64);

    const total_words_size = try co_matrix.sumAll();
    defer total_words_size.deinit();
    const t_w_s = try total_words_size.dataItem();

    const total_v = @as(f64, @floatFromInt(t_w_s));

    // co_matrix_f64.mulScalar_(2.0);
    co_matrix_f64.mulScalar_(total_v);

    const cols_words = try co_matrix.sum(0);

    var cm_iter = co_matrix.shapeIter();
    while (cm_iter.next()) |idx| {
        const one_c = try cols_words.getData([2]usize{ 0, idx[0] });
        const two_c = try cols_words.getData([2]usize{ 0, idx[1] });

        const v = co_matrix_f64.getData(idx) catch unreachable;

        std.debug.print("idx: {any} v= {} o= {} t= {}\n", .{ idx, v, one_c, two_c });

        const res_v = @log2(
            v /
                @as(f64, @floatFromInt(one_c)) /
                @as(f64, @floatFromInt(two_c)) + 1e-8,
        );
        try co_matrix_f64.setData(
            idx,
            @max(0.0, res_v),
        );
    }

    return co_matrix_f64;
}

test "text preprocess" {
    const allocator = std.testing.allocator;
    const text = "You say goodbye and I say hello.";

    var arena_act = std.heap.ArenaAllocator.init(allocator);
    defer arena_act.deinit();

    const ac = arena_act.allocator();

    const result = try preprocess(ac, text[0..]);
    const corpus = result.corpus;
    const word_to_id = result.word_to_id;
    const id_to_word = result.id_to_word;

    for (corpus.items) |cp| {
        std.debug.print("{} ", .{cp});
    }
    std.debug.print("\n", .{});

    var w_it = word_to_id.iterator();
    while (w_it.next()) |entry| {
        std.debug.print("{s}={} ", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
    std.debug.print("\n", .{});

    var i_it = id_to_word.iterator();
    while (i_it.next()) |entry| {
        std.debug.print("{}={s} ", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
    std.debug.print("\n", .{});

    // const vocab_size = SizeExpr.

    var shape_env = try ShapeEnv.init(ac);
    defer shape_env.deinit();

    const vocab_size = comptime SizeExpr.sym(.{ .name = "vocab_size" });
    try shape_env.bind(&vocab_size.Sym, word_to_id.count());

    const co_m = try create_co_matrix(
        ac,
        &shape_env,
        &corpus,
        vocab_size,
        1,
    );
    std.debug.print("co_m: {f}\n", .{co_m});

    const c0_s = "you";
    const c1_s = "i";
    const c0_se = comptime SizeExpr.sym(.{ .name = c0_s });
    const c1_se = comptime SizeExpr.sym(.{ .name = c1_s });
    try shape_env.bind(&c0_se.Sym, word_to_id.get(c0_s).?);
    try shape_env.bind(&c1_se.Sym, word_to_id.get(c1_s).?);

    const c0 = try co_m.sliceView(&.{tensor.SliceExpr.index(c0_se)});
    const c1 = try co_m.sliceView(&.{tensor.SliceExpr.index(c1_se)});

    std.debug.print("c0: {f}\n", .{c0});
    std.debug.print("c1: {f}\n", .{c1});

    const cos_sim = try c0.cosineSimilarity(&c1);
    std.debug.print("cos_sim: {}\n", .{cos_sim});

    try mostSimilar(
        allocator,
        &shape_env,
        "you",
        vocab_size,
        &co_m,
        &word_to_id,
        5,
    );

    const res = try ppmi(vocab_size, &co_m);
    std.debug.print("res: {f}\n", .{res});
}

test {
    std.testing.refAllDecls(@This());
}
