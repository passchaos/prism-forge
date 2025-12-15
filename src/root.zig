//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const Tensor = @import("Tensor.zig");
const DataType = @import("dtype.zig").DataType;
const F = @import("nn/functional.zig");

pub fn bufferedPrint() !void {
    // Stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush(); // Don't forget to flush!
}

test "one_hot" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    // const F =

    {
        const t1 = try Tensor.arange(allocator, DataType.i32, .{ .end = 10, .step = 3 });

        const t2 = try F.oneHot(t1, .{});

        const arr1 = [4][10]i32{
            .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            .{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
            .{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
            .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        };

        const r1 = try Tensor.fromShapedData(allocator, &arr1);
        try std.testing.expect(t2.equal(&r1));
        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
    }

    {
        var t1 = try Tensor.arange(allocator, DataType.i32, .{ .end = 10, .step = 3 });
        try t1.reshape_(&.{ 2, 2 });

        const t2 = try F.oneHot(t1, .{ .num_classes = 15 });

        std.debug.print("t2: {f}\n", .{t2});

        const arr1 = [2][2][15]i32{
            .{ .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, .{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
            .{ .{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 } },
        };

        const r1 = try Tensor.fromShapedData(allocator, &arr1);

        try std.testing.expect(t2.equal(&r1));
        std.debug.print("r1: {f} t2: {f}\n", .{ r1, t2 });
    }
}

test "pad" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    {
        const t1 = try Tensor.arange(allocator, DataType.i32, .{ .end = 10, .step = 3 });
        const t2 = try F.pad(t1, &.{ 1, 2 }, 100);
        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });
        const arr1 = [7]i32{ 100, 0, 3, 6, 9, 100, 100 };
        const r1 = try Tensor.fromShapedData(allocator, &arr1);
        try std.testing.expect(t2.equal(&r1));
        // std.debug.print("r1: {f} t2: {f}\n", .{ r1, t2 });
    }

    {
        var t1 = try Tensor.arange(allocator, DataType.i32, .{ .end = 10, .step = 3 });
        try t1.reshape_(&.{ 2, 2 });

        const t2 = try F.pad(t1, &.{ 1, 2, 3, 2 }, 100);
        std.debug.print("t1: {f} t2: {f}\n", .{ t1, t2 });

        const arr1 = [7][5]i32{
            .{ 100, 100, 100, 100, 100 },
            .{ 100, 100, 100, 100, 100 },
            .{ 100, 100, 100, 100, 100 },
            .{ 100, 0, 3, 100, 100 },
            .{ 100, 6, 9, 100, 100 },
            .{ 100, 100, 100, 100, 100 },
            .{ 100, 100, 100, 100, 100 },
        };
        const r1 = try Tensor.fromShapedData(allocator, &arr1);
        try std.testing.expect(t2.equal(&r1));
        // std.debug.print("r1: {f} t2: {f}\n", .{ r1, t2 });
    }
}
