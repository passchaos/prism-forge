//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const Tensor = @import("Tensor.zig");
const DataType = @import("dtype.zig").DataType;
const F = @import("nn/functional.zig");
const B = @import("nn/basic.zig");

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

test "mnist" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const env_home = std.posix.getenv("HOME").?;

    const path = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/t10k-images.idx3-ubyte" });
    const file = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });

    var buf = [_]u8{0} ** 4;

    const nm = try file.read(buf[0..]);
    const magic_number = std.mem.readInt(u32, &buf, .big);

    const ns = try file.read(buf[0..]);
    const samples = std.mem.readInt(u32, &buf, .big);

    const nr = try file.read(buf[0..]);
    const rows = std.mem.readInt(u32, &buf, .big);

    const nc = try file.read(buf[0..]);
    const columns = std.mem.readInt(u32, &buf, .big);

    const data_size = samples * rows * columns;
    const data = try file.readToEndAlloc(allocator, data_size);

    std.debug.print("read n: {} magic number: {} ns: {} samples: {} nr: {} rows: {} nc: {} columns: {}\n", .{ nm, magic_number, ns, samples, nr, rows, nc, columns });
    std.debug.print("data size: {}\n", .{data.len});
}

test "loss" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const a = try Tensor.rand(allocator, f32, &.{ 2, 5 }, -1.0, 1.0);

    const a1 = try a.softmax();
    const b = try Tensor.rand(allocator, f32, &.{ 2, 5 }, 0.0, 1.0);
    const b1 = try b.softmax();

    // std.debug.print("a: {f} a1: {f} b: {f} b1: {f}\n", .{ a, a1, b, b1 });

    const mse_loss = try F.mseLoss(a1, b1);
    const cross_entropy = try F.crossEntropy(a1, b1);

    std.debug.print("mse_loss: {f} cross_entropy: {f}\n", .{ mse_loss, cross_entropy });
}

fn Fh(T: type) type {
    return struct {
        fn call(x: T) T {
            return 0.01 * x * x + 0.1 * x;
        }
    };
}

test "differential" {
    const Ft = f64;
    const fh = Fh(Ft);

    for ([2]Ft{ 5.0, 10.0 }) |x| {
        const df1 = B.numericalDiff(x, fh.call);
        std.debug.print("df1: {}\n", .{df1});
    }
}
