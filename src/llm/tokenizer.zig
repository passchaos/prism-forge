const std = @import("std");

const Rule = struct {
    token: []const u8,
    merge: []const u8,
};

const BPETokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringArrayHashMap(usize),
    merges: std.ArrayList(Rule),

    const Self = @This();

    fn deinit(self: *Self) void {
        self.vocab.deinit();
        self.merges.deinit(self.allocator);
    }

    fn new(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .vocab = std.StringArrayHashMap(usize).init(allocator),
            .merges = try std.ArrayList(Rule).initCapacity(allocator, 10),
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
            std.debug.print("char: {s}\n", .{char});
            try tokens.append(self.allocator, char);
        }
        std.debug.print("{any}", .{tokens});
    }

    // fn encode(self: *Self, text: []const u8) !std.ArrayList(usize) {}

    // fn decode(self: *Self, tokens: []const usize) !std.ArrayList(u8) {}
};

test "bpe tokenizer" {
    const allocator = std.testing.allocator;

    var token = try BPETokenizer.new(allocator);
    defer token.deinit();

    try token.train("you are bigger", 300);
}
