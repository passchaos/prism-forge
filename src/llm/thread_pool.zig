const std = @import("std");

const Bt = struct {
    const Self = @This();
    fn deinit(self: *Self) void {
        self.pool.deinit();
    }

    pool: std.Thread.Pool = undefined,

    fn init(self: *Self, gpa: std.mem.Allocator) !void {
        var pool_i: std.Thread.Pool = undefined;

        try pool_i.init(.{
            .allocator = gpa,
            .track_ids = true,
        });

        self.pool = pool_i;
    }
    fn initSuccess(self: *Self, gpa: std.mem.Allocator) !void {
        try self.pool.init(.{
            .allocator = gpa,
            .track_ids = true,
        });
    }
    fn new(gpa: std.mem.Allocator) !Self {
        var val = Self{};
        try val.initSuccess(gpa); // 调用修复后的 init
        return val;
    }
    fn new2(gpa: std.mem.Allocator) !Self {
        var pool_i: std.Thread.Pool = undefined;

        try pool_i.init(.{
            .allocator = gpa,
            .track_ids = true,
        });

        return Self{
            .pool = pool_i,
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // var bt = Bt{};
    // try bt.init(allocator);
    // try bt.initSuccess(allocator);
    var bt = try Bt.new2(allocator);
    // try bt.pool.init(.{ .allocator = allocator, .track_ids = true });

    // var pool1 = bt.pool;
    // defer bt.deinit();
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

        // if (idx == 4) {
        //     try pool1.spawn(func, .{10});
        // }
    }

    bt.pool.waitAndWork(&wg);
    // wg.wait();
}
