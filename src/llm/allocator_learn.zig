const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();
    // const allocator = gpa.allocator();

    const count = 100;
    const thre = 50;

    var bytes: [count][]u8 = undefined;

    for (0..count) |i| {
        if (i == thre) {
            std.debug.print("begin free wait\n", .{});
            std.Thread.sleep(20_000_000_000);
            std.debug.print("begin release\n", .{});
        } else if (i > thre) {
            allocator.free(bytes[i - thre]);
        } else {
            bytes[i] = try allocator.alloc(u8, 100_000_000);
        }
    }

    std.debug.print("after release\n", .{});
    std.Thread.sleep(10_000_000_000);

    std.debug.print("finish\n", .{});
}
