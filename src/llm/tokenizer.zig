const std = @import("std");

const BPETokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringArrayHashMap(usize),
    merges: std.ArrayList(struct { []const u8, []const u8 }),

    const Self = @This();

    fn deinit(self: *Self) void {
        self.vocab.deinit();
        self.merges.deinit(self.allocator);
    }

    fn new(allocator: std.mem.Allocator) !Self {
        var vocab = std.StringArrayHashMap(usize).init(allocator);

        for (0..256) |i| {
            const char = try std.fmt.allocPrint(allocator, "<{x:0>2}>", .{i});
            try vocab.put(char, i);
        }

        return Self{
            .allocator = allocator,
            .vocab = vocab,
            .merges = try std.ArrayList(struct { []const u8, []const u8 }).initCapacity(allocator, 10),
        };
    }

    fn train(self: *Self, text: []const u8, vocab_size: usize) !void {
        if (vocab_size <= 256) {
            return;
        }

        std.debug.print("Training BPE tokenizer...", .{});
        std.debug.print("  Starting..", .{});

        var tokens = try std.ArrayList([]const u8)
            .initCapacity(self.allocator, text.len);
        defer {
            for (tokens.items) |token| {
                self.allocator.free(token);
            }
            tokens.deinit(self.allocator);
        }

        for (text) |byte| {
            const char = try std.fmt.allocPrint(self.allocator, "<{x:0>2}>", .{byte});
            // std.debug.print("char: {s}\n", .{char});
            try tokens.append(self.allocator, char);
        }
        // std.debug.print("{any}\n", .{tokens});

        for (256..vocab_size) |idx| {
            if (tokens.items.len == 0) {
                break;
            }

            var pair_maps = std.StringArrayHashMap(std.ArrayList(usize))
                .init(self.allocator);

            for (0..tokens.items.len - 1) |i| {
                const pair_key = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ tokens.items[i], tokens.items[i + 1] });

                // std.debug.print("pair_key: {s}\n", .{pair_key});
                var entry = try pair_maps.getOrPut(pair_key);

                if (entry.found_existing) {
                    try entry.value_ptr.append(self.allocator, i);
                } else {
                    entry.value_ptr.* = try std.ArrayList(usize).initCapacity(self.allocator, 10);
                    try entry.value_ptr.append(self.allocator, i);
                }
            }

            const PairIndexes = struct {
                pair: []const u8,
                indexes: std.ArrayList(usize),

                fn lessThan(_: void, a: @This(), b: @This()) bool {
                    return a.indexes.items.len < b.indexes.items.len;
                }
            };

            var pair_maps_sorted_list = try std.ArrayList(PairIndexes).initCapacity(self.allocator, pair_maps.count());

            var pm_iter = pair_maps.iterator();
            while (pm_iter.next()) |entry| {
                const pair = entry.key_ptr.*;
                const indexes = entry.value_ptr.*;
                try pair_maps_sorted_list.append(self.allocator, PairIndexes{ .pair = pair, .indexes = indexes });
            }
            std.mem.sort(
                PairIndexes,
                pair_maps_sorted_list.items,
                void{},
                PairIndexes.lessThan,
            );

            const top_pair_indexes = pair_maps_sorted_list.items[0];

            std.debug.print("add vocab: key= {s} idx= {}\n", .{ top_pair_indexes.pair, idx });
            try self.vocab.put(top_pair_indexes.pair, idx);

            var i = top_pair_indexes.indexes.items.len - 1;

            {
                const one_idx = top_pair_indexes.indexes.items[i];

                std.debug.print("merge: {s}{s}\n", .{ tokens.items[one_idx], tokens.items[one_idx + 1] });
                try self.merges.append(
                    self.allocator,
                    .{ tokens.items[one_idx], tokens.items[one_idx + 1] },
                );
            }

            while (i >= 0) : (i -= 1) {
                const token_idx = top_pair_indexes.indexes.items[i];
                tokens.orderedRemoveMany(&.{ token_idx, token_idx + 1 });

                if (i == 0) {
                    break;
                }
            }
        }
    }

    // fn encode(self: *Self, text: []const u8) !std.ArrayList(usize) {}

    // fn decode(self: *Self, tokens: []const usize) !std.ArrayList(u8) {}
};

test "bpe tokenizer" {
    const allocator = std.testing.allocator;

    var arena_alloc = std.heap.ArenaAllocator.init(allocator);
    defer arena_alloc.deinit();
    const arena_allocator = arena_alloc.allocator();

    var token = try BPETokenizer.new(arena_allocator);
    // defer token.deinit();

    try token.train("you are bigger", 300);
}
