const std = @import("std");
const zeit = @import("zeit");

var logger: ?Logger = null;

const Logger = struct {
    allocator: std.mem.Allocator,
    tz: zeit.TimeZone,

    const Self = @This();

    fn deinit(self: *Self) void {
        self.tz.deinit();
    }

    fn init(allocator: std.mem.Allocator) !Self {
        var env_map = try std.process.getEnvMap(allocator);
        defer env_map.deinit();

        const tz = try zeit.local(allocator, &env_map);

        return Self{
            .allocator = allocator,
            .tz = tz,
        };
    }

    fn print(_: *const Self, comptime src: std.builtin.SourceLocation, comptime format: []const u8, args: anytype) !void {
        const utc = try zeit.instant(.{});
        // const local = utc.in(&self.tz);

        const time_t = utc.time();

        var buf: [64]u8 = undefined;
        const written = try time_t.bufPrint(&buf, .rfc3339Nano);

        std.debug.print("{s} {s} {s}:{d} {s} " ++ format, .{ written, src.module, src.file, src.line, src.fn_name } ++ args);
    }
};

fn init() !void {
    var buf: [1024 * 12]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const allocator = fba.allocator();

    logger = try Logger.init(allocator);
}

pub fn print(comptime src: std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
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
