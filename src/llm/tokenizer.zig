const std = @import("std");

var pool: std.Thread.Pool = undefined;

const ByteCodedContent = struct {
    allocator: std.mem.Allocator,
    inner: std.ArrayList([]const u8),

    const Self = @This();

    fn deinit(self: *Self) void {
        for (self.inner.items) |item| {
            self.allocator.free(item);
        }

        self.inner.deinit(self.allocator);
    }

    pub fn format(self: Self, writer: *std.Io.Writer) std.io.Writer.Error!void {
        for (self.inner.items) |item| {
            _ = try writer.write(" ");
            _ = try writer.write(item);
        }
    }
};

fn byteEncoded(allocator: std.mem.Allocator, text: []const u8) !ByteCodedContent {
    var tokens = try std.ArrayList([]const u8)
        .initCapacity(allocator, text.len);

    for (text) |byte| {
        const char = try std.fmt.allocPrint(allocator, "<{x:0>2}>", .{byte});
        // std.debug.print("char: {s}\n", .{char});
        try tokens.append(allocator, char);
    }

    return ByteCodedContent{
        .allocator = allocator,
        .inner = tokens,
    };
}

const BPETokenizer = struct {
    arena: *std.heap.ArenaAllocator,
    vocab: std.StringArrayHashMap(usize),
    merges: std.ArrayList(struct { []const u8, []const u8 }),
    pool: std.Thread.Pool = undefined,

    const Self = @This();

    fn deinit(self: *Self) void {
        _ = self.arena.reset(.free_all);
    }

    fn new(arena: *std.heap.ArenaAllocator) !Self {
        const allocator = arena.allocator();
        var vocab = std.StringArrayHashMap(usize)
            .init(allocator);

        var pool_i: std.Thread.Pool = undefined;
        try pool_i.init(.{
            .allocator = allocator,
        });
        // var gpa1 = std.heap.GeneralPurposeAllocator(.{}){};

        for (0..256) |i| {
            const char = try std.fmt.allocPrint(allocator, "<{x:0>2}>", .{i});
            try vocab.put(char, i);
        }

        return Self{
            .arena = arena,
            .vocab = vocab,
            .merges = try std.ArrayList(struct { []const u8, []const u8 })
                .initCapacity(allocator, 10),
            .pool = pool_i,
        };
    }

    fn train(self: *Self, text: []const u8, vocab_size: usize) !void {
        const alloc = self.arena.allocator();
        var thread_safe_wrapper = std.heap.ThreadSafeAllocator{
            .child_allocator = alloc,
        };
        const allocator = thread_safe_wrapper.allocator();

        if (vocab_size <= 256) {
            return;
        }

        std.debug.print("Training BPE tokenizer...\n", .{});
        std.debug.print("  Starting.. {}\n", .{try std.Thread.getCpuCount()});

        var tokens = try byteEncoded(allocator, text);
        defer tokens.deinit();

        // std.debug.print("{any}\n", .{tokens});

        const part_count: usize = 10;
        var part_arenas: [part_count]std.heap.ArenaAllocator = undefined;
        for (0..part_count) |part_idx| {
            var gpa = std.heap.GeneralPurposeAllocator(.{}){};
            const gpa_alloc = gpa.allocator();
            const arena = std.heap.ArenaAllocator.init(gpa_alloc);
            part_arenas[part_idx] = arena;
        }
        defer {
            for (part_arenas) |part_arena| {
                part_arena.deinit();
            }
        }

        for (256..vocab_size) |idx| {
            if (tokens.inner.items.len <= 1) {
                break;
            }

            const start_ts = try std.time.Instant.now();

            var pair_maps: std.StringArrayHashMap(std.ArrayList(usize)) = undefined;

            {
                const par_start_ts = try std.time.Instant.now();
                const part_size = (tokens.inner.items.len + part_count - 1) / part_count;

                var work_group = std.Thread.WaitGroup{};

                var part_maps = try allocator.alloc(
                    std.StringArrayHashMap(std.ArrayList(usize)),
                    part_count,
                );

                for (0..part_count) |part_idx| {
                    const chunk_start = part_size * part_idx;
                    const chunk_end = @min(part_size * (part_idx + 1), tokens.inner.items.len);

                    const part_alloc = part_arenas[part_idx].allocator();

                    part_maps[part_idx] = std.StringArrayHashMap(std.ArrayList(usize)).init(part_alloc);

                    // std.debug.print("begin spawn wg\n", .{});
                    self.pool.spawnWg(&work_group, struct {
                        fn funcImpl(s_alloc: std.mem.Allocator, map: *std.StringArrayHashMap(std.ArrayList(usize)), tokens_items: [][]const u8, range_start: usize, range_end: usize) !void {
                            for (range_start..range_end) |i| {
                                if (i == tokens_items.len - 1) continue;
                                const pair_key = try std.fmt.allocPrint(
                                    s_alloc,
                                    "{s}{s}",
                                    .{ tokens_items[i], tokens_items[i + 1] },
                                );
                                // std.debug.print("pair_key: {s}\n", .{pair_key});

                                // std.debug.print("pair_key: {s}\n", .{pair_key});
                                const entry = try map.getOrPut(pair_key);

                                if (entry.found_existing) {
                                    try entry.value_ptr.append(s_alloc, i);
                                    // entry.value_ptr.* += 1;
                                } else {
                                    var value = try std.ArrayList(usize).initCapacity(s_alloc, 10);
                                    try value.append(s_alloc, i);
                                    entry.value_ptr.* = value;
                                }
                            }

                            // std.Thread.sleep(10000000000);
                        }

                        fn func(s_alloc: std.mem.Allocator, map: *std.StringArrayHashMap(std.ArrayList(usize)), tokens_items: [][]const u8, range_start: usize, range_end: usize) void {
                            funcImpl(s_alloc, map, tokens_items, range_start, range_end) catch unreachable;
                        }
                    }.func, .{ part_alloc, &part_maps[part_idx], tokens.inner.items, chunk_start, chunk_end });

                    // const part_map = try std.StringArrayHashMap(std.ArrayList(usize)).initCapacity(allocator, chunk.len);
                    // part_maps.append(part_map);
                }

                self.pool.waitAndWork(&work_group);

                const par_part_ts = try std.time.Instant.now();

                pair_maps = part_maps[0];

                const pair_maps_alloc = part_arenas[0].allocator();

                for (1..part_count) |part_idx| {
                    var part_iter = part_maps[part_idx].iterator();

                    while (part_iter.next()) |entry| {
                        const key = entry.key_ptr.*;
                        const value = entry.value_ptr.*;

                        const target_entry_res = try pair_maps.getOrPut(key);
                        if (target_entry_res.found_existing) {
                            try target_entry_res.value_ptr.appendSlice(pair_maps_alloc, value.items);
                            // target_entry_res.value_ptr.* += value;
                        } else {
                            var target_value = try std.ArrayList(usize).initCapacity(pair_maps_alloc, value.items.len);
                            try target_value.appendSlice(pair_maps_alloc, value.items);
                            target_entry_res.value_ptr.* = value;
                        }
                    }
                }

                const reduce_ts = try std.time.Instant.now();
                std.debug.print(
                    "part dur: {} reduce dur: {}\n",
                    .{ par_part_ts.since(par_start_ts) / 1000, reduce_ts.since(par_part_ts) / 1000 },
                );
            }

            // pair_maps = std.StringArrayHashMap(usize)
            //     .init(allocator);

            // for (0..tokens.inner.items.len - 1) |i| {
            //     const pair_key = try std.fmt.allocPrint(
            //         allocator,
            //         "{s}{s}",
            //         .{ tokens.inner.items[i], tokens.inner.items[i + 1] },
            //     );

            //     // std.debug.print("pair_key: {s}\n", .{pair_key});
            //     const entry = try pair_maps.getOrPut(pair_key);

            //     if (entry.found_existing) {
            //         entry.value_ptr.* += 1;
            //     } else {
            //         entry.value_ptr.* = 1;
            //     }
            // }

            const pair_construction_duration = try std.time.Instant.now();
            // std.debug.print("  Pair construction: {any}\n", .{pair_construction_duration});

            const PairIndexes = struct {
                pair: []const u8,
                indexes: std.ArrayList(usize),
                // count: usize,

                fn lessThan(_: void, a: @This(), b: @This()) bool {
                    return a.indexes.items.len < b.indexes.items.len;
                    // return a.count < b.count;
                }

                fn greatThan(_: void, a: @This(), b: @This()) bool {
                    return a.indexes.items.len > b.indexes.items.len;
                    // return a.count > b.count;
                }
            };

            var pair_maps_sorted_list = try std.ArrayList(PairIndexes)
                .initCapacity(allocator, pair_maps.count());

            var pm_iter = pair_maps.iterator();
            while (pm_iter.next()) |entry| {
                const pair = entry.key_ptr.*;
                const count = entry.value_ptr.*;

                // std.debug.print("pair: {s} indexes: {any}\n", .{ pair, indexes });
                try pair_maps_sorted_list.append(
                    allocator,
                    PairIndexes{ .pair = pair, .indexes = count },
                );
            }
            std.mem.sort(
                PairIndexes,
                pair_maps_sorted_list.items,
                void{},
                PairIndexes.greatThan,
            );
            const pair_sorted_time = try std.time.Instant.now();
            // std.debug.print("  Pair construction: {any}\n", .{pair_construction_duration});
            // std.debug.print("  Pair sorted: {any}\n", .{pair_sorted_time});

            // for (pair_maps_sorted_list.items) |pair_indexes| {
            //     std.debug.print("pair: {s} indexes: {any}\n", .{ pair_indexes.pair, pair_indexes.indexes.items.len });
            // }

            const top_pair_indexes = pair_maps_sorted_list.items[0];
            // const top_pair = pair_maps_sorted_list.items[0];

            std.debug.print("add vocab: key= {s} idx= {}\n", .{ top_pair_indexes.pair, idx });

            // var self_vocab_iter = self.vocab.iterator();
            // while (self_vocab_iter.next()) |entry| {
            //     std.debug.print("vocab: key= {s} idx= {}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            // }

            const top_pair_global_key = try allocator.alloc(u8, top_pair_indexes.pair.len);
            @memcpy(top_pair_global_key, top_pair_indexes.pair);
            try self.vocab.put(top_pair_global_key, idx);
            // try self.merges.append(allocator, .{top}})

            var i = top_pair_indexes.indexes.items.len - 1;
            std.debug.print("indexes: {any}\n", .{top_pair_indexes.indexes.items.len});

            {
                const last_idx = top_pair_indexes.indexes.items[i];

                const b_t = tokens.inner.orderedRemove(last_idx + 1);

                const a_t = tokens.inner.items[last_idx];
                tokens.inner.items[last_idx] = top_pair_indexes.pair;

                const a_t_g = try allocator.alloc(u8, a_t.len);
                @memcpy(a_t_g, a_t);
                const b_t_g = try allocator.alloc(u8, b_t.len);
                @memcpy(b_t_g, b_t);
                // std.debug.print("merge: {s}{s}\n", .{ a_t, b_t });
                try self.merges.append(
                    allocator,
                    .{ a_t_g, b_t_g },
                );

                if (i == 0) {
                    continue;
                }

                i -= 1;
            }

            // // const begin = std.time.milliTimestamp();

            var remove_indexes = try std.ArrayList(usize).initCapacity(allocator, i);

            while (i >= 0) {
                const token_idx = top_pair_indexes.indexes.items[i];

                try remove_indexes.append(allocator, token_idx + 1);
                // std.debug.print("token_idx: {} i= {}\n", .{ token_idx, i });
                // _ = tokens.inner.orderedRemove(token_idx + 1);

                tokens.inner.items[token_idx] = top_pair_indexes.pair;

                if (i == 0) {
                    // const end = std.time.milliTimestamp();
                    // std.debug.print("merge time: {}ms\n", .{(end - begin)});
                    break;
                } else {
                    i -= 1;
                }
            }

            std.mem.sort(usize, remove_indexes.items, void{}, struct {
                fn func(_: void, a: usize, b: usize) bool {
                    return a < b;
                }
            }.func);

            tokens.inner.orderedRemoveMany(remove_indexes.items);

            const final_ts = try std.time.Instant.now();
            std.debug.print("pair construction: {}\n", .{pair_construction_duration.since(start_ts) / 1000});
            std.debug.print("pair sort: {}\n", .{pair_sorted_time.since(pair_construction_duration) / 1000});
            std.debug.print("final dur: {}\n", .{final_ts.since(pair_sorted_time) / 1000});
            // std.debug.print("  Pair construction: {any}\n", .{pair_construction_duration});
            // std.debug.print("  Pair sorted: {any}\n", .{pair_sorted_time});
            // std.debug.print("  Final: {any}\n", .{final_ts});
        }
    }

    fn encode(self: *const Self, text: []const u8) !std.ArrayList(usize) {
        // defer {
        //     _ = self.arena.reset(.free_all);
        // }
        const allocator = self.arena.allocator();

        var tokens = try byteEncoded(allocator, text);
        defer tokens.deinit();
        std.debug.print("tokens: {f}\n", .{tokens});

        for (self.merges.items) |merge| {
            const a = merge[0];
            const b = merge[1];

            if (tokens.inner.items.len > 1) {
                var new_tokens = try std.ArrayList([]const u8)
                    .initCapacity(allocator, tokens.inner.items.len);

                var i: usize = 0;
                std.debug.print("tokens len: {}\n", .{tokens.inner.items.len});
                while (i < tokens.inner.items.len) {
                    std.debug.print("i: {}\n", .{i});
                    if (i < tokens.inner.items.len - 1 and std.mem.eql(u8, a, tokens.inner.items[i]) and
                        std.mem.eql(u8, b, tokens.inner.items[i + 1]))
                    {
                        const merged = try std.fmt.allocPrint(allocator, "{s}{s}", .{ a, b });
                        try new_tokens.append(allocator, merged);
                        i += 2;
                    } else {
                        try new_tokens.append(allocator, tokens.inner.items[i]);
                        i += 1;
                    }
                }

                std.debug.print("new_tokens:", .{});
                for (new_tokens.items) |token| {
                    std.debug.print(" {s}", .{token});
                }
                std.debug.print("\n", .{});

                tokens.inner.deinit(allocator);
                tokens.inner = new_tokens;
            }
        }

        var result = try std.ArrayList(usize).initCapacity(allocator, tokens.inner.items.len);
        for (tokens.inner.items) |token| {
            const token_id = self.vocab.get(token).?;
            try result.append(allocator, token_id);
        }

        // std.debug.print("{any}\n", .{tokens});

        return result;
    }

    fn decode(self: *Self, tokens: []const usize) ![]u8 {
        const allocator = self.arena.allocator();

        var id_to_token: std.AutoHashMap(usize, []const u8) = std.AutoHashMap(usize, []const u8).init(allocator);

        var vocab_iter = self.vocab.iterator();
        while (vocab_iter.next()) |entry| {
            const token = entry.key_ptr.*;
            const token_id = entry.value_ptr.*;

            try id_to_token.put(token_id, token);
        }

        var result = try std.ArrayList(u8).initCapacity(allocator, 10);
        for (tokens) |token_id| {
            const token = id_to_token.get(token_id).?;

            var i: usize = 0;
            while (i < token.len) : (i += 4) {
                const hex_str = token[i + 1 .. i + 3];

                const byte = try std.fmt.parseInt(u8, hex_str, 16);
                try result.append(allocator, byte);
            }
        }
        return result.toOwnedSlice(allocator);
    }
};

test "bpe tokenizer" {
    const allocator = std.testing.allocator;

    var arena_alloc = std.heap.ArenaAllocator.init(allocator);
    defer arena_alloc.deinit();

    var token = try BPETokenizer.new(&arena_alloc);
    defer token.deinit();

    try token.train("hello hello world world hello", 300);

    const res = try token.encode("hello world");
    std.debug.print("res: {any}\n", .{res});

    const decoded = try token.decode(res.items);
    std.debug.print("decoded: {s}\n", .{decoded});
}

test "gutenberg" {
    std.debug.print("is single threaded: {}\n", .{@import("builtin").single_threaded});
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // var gpa_alloc = gpa.allocator();

    var arena_alloc = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_alloc.deinit();

    const allocator = arena_alloc.allocator();

    var token = try BPETokenizer.new(&arena_alloc);
    defer token.deinit();

    const file = try std.fs.openFileAbsolute("/tmp/pg100.txt", .{ .mode = .read_only });
    var file_buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&file_buffer);

    const file_size = try file_reader.getSize();

    const file_contents = try file_reader.interface.readAlloc(allocator, @intCast(file_size));

    std.debug.print("file_size: {}\n", .{file_size});
    try token.train(file_contents, 512);
}

fn threadPoolTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa_alloc = gpa.allocator();

    // var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = gpa_alloc });
    defer pool.deinit();

    var wg: std.Thread.WaitGroup = .{};

    const func = struct {
        fn run() void {
            std.debug.print("thread id: {}\n", .{std.Thread.getCurrentId()});
            std.Thread.sleep(1000000000000);
            std.debug.print("thread finish: {}\n", .{std.Thread.getCurrentId()});
        }
    }.run;

    for (0..5) |_| {
        pool.spawnWg(&wg, func, .{});
    }

    pool.waitAndWork(&wg);
}

pub fn main() !void {
    // try threadPoolTest();
    std.debug.print("is single threaded: {}\n", .{@import("builtin").single_threaded});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa_alloc = gpa.allocator();

    var arena_alloc = std.heap.ArenaAllocator.init(gpa_alloc);
    defer arena_alloc.deinit();

    // const allocator = arena_alloc.allocator();

    try pool.init(.{ .allocator = gpa_alloc });
    // defer pool.deinit();

    var token = try BPETokenizer.new(&arena_alloc);
    defer token.deinit();

    // var pool = token.pool;

    // var pool: std.Thread.Pool = undefined;

    // var wg: std.Thread.WaitGroup = .{};

    // const func = struct {
    //     fn run() void {
    //         std.debug.print("thread id: {}\n", .{std.Thread.getCurrentId()});
    //         std.Thread.sleep(1000000000000);
    //         std.debug.print("thread finish: {}\n", .{std.Thread.getCurrentId()});
    //     }
    // }.run;

    // for (0..5) |_| {
    //     pool.spawnWg(&wg, func, .{});
    // }

    // pool.waitAndWork(&wg);

    const file = try std.fs.openFileAbsolute("/tmp/pg100.txt", .{ .mode = .read_only });
    var file_buffer: [1024]u8 = undefined;
    var file_reader = file.reader(&file_buffer);

    const file_size = try file_reader.getSize();

    const file_contents = try file_reader.interface.readAlloc(gpa_alloc, @intCast(file_size));

    std.debug.print("file_size: {}\n", .{file_size});
    try token.train(file_contents, 512);
}
