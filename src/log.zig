const std = @import("std");
const zeit = @import("zeit");
const zdt = @import("zdt");

var logger: ?Logger = null;
var rwlock: std.Thread.RwLock = std.Thread.RwLock{};

const Logger = struct {
    allocator: std.mem.Allocator,
    tz: zdt.Timezone,
    // tz: zeit.TimeZone,

    const Self = @This();

    fn deinit(self: *Self) void {
        self.tz.deinit();
    }

    fn init(allocator: std.mem.Allocator) !Self {
        // var env_map = try std.process.getEnvMap(allocator);
        // defer env_map.deinit();

        const tz = try zdt.Timezone.tzLocal(allocator);
        // const tz = try zeit.local(allocator, &env_map);

        return Self{
            .allocator = allocator,
            .tz = tz,
        };
    }

    fn print(self: *const Self, comptime src: std.builtin.SourceLocation, comptime format: []const u8, args: anytype) !void {
        const local = try zdt.Datetime.now(zdt.Datetime.tz_options{ .tz = &self.tz });
        // const utc = try zeit.instant(.{});
        // const local = utc.in(&self.tz);

        var buf = std.mem.zeroes([64]u8);
        var w = std.Io.Writer.fixed(&buf);
        try local.toString(zdt.Formats.RFC3339nano, &w);
        // const written = try time_t.bufPrint(&buf, .rfc3339Nano);

        std.debug.print("{s} {s} {s}:{d} {s} " ++ format, .{ buf, src.module, src.file, src.line, src.fn_name } ++ args);
    }
};

fn init() !void {
    var buf: [1024 * 12]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const allocator = fba.allocator();

    std.debug.print("current thread: {}\n", .{std.Thread.getCurrentId()});
    // rwlock.lock();
    logger = try Logger.init(allocator);
    // rwlock.unlock();
}

pub fn print(comptime src: std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
    // rwlock.lockShared();

    if (logger == null) {
        std.debug.print("init logger\n", .{});
        init() catch |err| {
            std.debug.print("init meet error: {}\n", .{err});
        };
    }

    if (logger) |lg| {
        lg.print(src, format, args) catch |err| {
            std.debug.print("meet error: {}\n", .{err});
        };
    }
}
