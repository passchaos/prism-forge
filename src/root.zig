//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const tensor = @import("tensor.zig");
pub const DType = tensor.DataType;
pub const Tensor = tensor.Tensor;

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

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "tensor creation" {
    const TensorF32x3x2 = Tensor(DType.f32, &.{ 3, 2 });

    const t1 = TensorF32x3x2.init(0.1);
    std.debug.print("t1: {f}\n", .{t1});

    const TensorU32x4x4x8 = Tensor(DType.u32, &.{ 4, 4, 8 });
    const t2 = TensorU32x4x4x8.init(77);
    std.debug.print("t2: {f}\n", .{t2});
    // _ = t1;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}
