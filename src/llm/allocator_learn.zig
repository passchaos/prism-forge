const std = @import("std");

test "div" {
    const a: isize = 10;
    const b: isize = 9;
    const d: isize = 3;
    const c: isize = -10;

    std.debug.print("int div\n", .{});

    std.debug.print("div 0: {} {} {}\n", .{ a / d, b / d, c / d });
    std.debug.print(
        "div trunc: {} {} {}\n",
        .{ @divTrunc(a, d), @divTrunc(b, d), @divTrunc(c, d) },
    );
    std.debug.print(
        "div floor: {} {} {}\n",
        .{ @divFloor(a, d), @divTrunc(b, d), @divFloor(c, d) },
    );
    std.debug.print(
        "div ceil: {} {} {}\n",
        .{
            try std.math.divCeil(isize, a, d),
            try std.math.divCeil(isize, b, d),
            try std.math.divCeil(isize, c, d),
        },
    );
}

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
