const std = @import("std");

const Bt = struct {
    pool: std.Thread.Pool = undefined,

    const Self = @This();
    fn deinit(self: *Self) void {
        self.pool.deinit();
    }
    fn new(gpa: std.mem.Allocator) !Self {
        // var pool_i: std.Thread.Pool = undefined;
        var val = Self{};
        try val.pool.init(.{
            .allocator = gpa,
            .track_ids = true,
        });

        return val;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var bt = try Bt.new(allocator);
    // try bt.pool.init(.{ .allocator = allocator, .track_ids = true });
    defer bt.deinit();
    // var pool = bt.pool;
    // var pool: std.Thread.Pool = undefined;
    // try pool.init(.{ .allocator = allocator, .track_ids = true });

    var wg = std.Thread.WaitGroup{};

    const func = struct {
        fn func(id: usize) void {
            std.debug.print("thread func begin: id= {}\n", .{id});
            std.Thread.sleep(100000000000);

            std.debug.print("thread func end: id= {}\n", .{id});
        }
    }.func;

    for (0..5) |_| {
        bt.pool.spawnWgId(&wg, func, .{});
    }

    wg.wait();
}
