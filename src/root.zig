//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const tensor = @import("tensor.zig");
const utils = @import("utils.zig");

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

// test "tensor creation" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     const allocator = gpa.allocator();

//     const TensorF32x3x2 = Tensor(DType.f32, &.{ 3, 2 });

//     const t1 = try TensorF32x3x2.init(allocator, .{}, 0.1);
//     defer t1.deinit(&allocator);
//     std.debug.print("t1: {f} data= {any}\n", .{ t1, t1.data });

//     const arr1 = [3][2]f32{
//         [2]f32{ 1.0, 2.0 },
//         [2]f32{ 3.0, 4.0 },
//         [2]f32{ 5.0, 6.0 },
//     };
//     const t11 = try TensorF32x3x2.from_data(allocator, .{}, &arr1);
//     std.debug.print("t11: {f}\n", .{t11});

//     const Tensor3U32x4x4x8 = Tensor(DType.u32, &.{ 4, 4, 8 });
//     const t2 = try Tensor3U32x4x4x8.init(allocator, .{}, 77);
//     defer t2.deinit(&allocator);
//     std.debug.print("t2: {f}\n", .{t2});
//     // _ = t1;

//     const Tensor3U32 = Tensor(DType.u32, &.{ null, null, null });
//     const t3 = try Tensor3U32.init(allocator, .{ .shape = .{ 4, 4, 8 } }, 21);
//     defer t3.deinit(&allocator);
//     std.debug.print("t3: {f}\n", .{t3});

//     const Tensor3U32_1 = Tensor(DType.u32, &.{ 3, null, 5 });
//     const t3_1 = try Tensor3U32_1.init(allocator, .{ .shape = .{ 4, 4, 8 } }, 21);
//     defer t3_1.deinit(&allocator);
//     std.debug.print("t3_1: {f}\n", .{t3_1});

//     const TensorU32 = Tensor(DType.u32, null);
//     const t4 = try TensorU32.init(allocator, .{ .shape = &.{ 1, 2, 3, 4 } }, 24);
//     defer t4.deinit(&allocator);
//     std.debug.print("t4: {f}\n", .{t4});

//     const t5 = try TensorU32.init(allocator, .{ .shape = &.{ 2, 3, 3, 14, 15 } }, 24);
//     defer t5.deinit(&allocator);
//     std.debug.print("t5: {f} {any}\n", .{ t5, t5._shape });
// }

// test "array structure" {
//     const a: [3][3]u32 = .{ .{ 1, 2, 3 }, .{ 3, 4, 5 }, .{ 5, 6, 7 } };
//     std.debug.print("a: {any}\n", .{@typeInfo(@TypeOf(a))});

//     const dims = utils.getDims(@TypeOf(a));
//     const data_len = utils.product(utils.getDims(@TypeOf(a)));
//     std.debug.print("dims: {any} {} {}\n", .{ dims, std.meta.Child(@TypeOf(a)), data_len });

//     const b = [_]usize{ 1, 2 };

//     const c: []const usize = b[0..];
//     const ta = @typeInfo(@TypeOf(&a));
//     const tb = @typeInfo(@TypeOf(&b));
//     const tc = @typeInfo(@TypeOf(c));
//     std.debug.print("ta: {any} tb: {any} tc: {any}\n", .{ ta, tb, tc });

//     switch (ta) {
//         .pointer => |p| {
//             std.debug.print("pointer: {any}\n", .{@typeInfo(p.child)});
//         },
//         else => {},
//     }

//     switch (tb) {
//         .pointer => |p| {
//             std.debug.print("pointer: {any}\n", .{@typeInfo(p.child)});
//         },
//         else => {},
//     }

//     switch (tc) {
//         .pointer => |p| {
//             std.debug.print("pointer: {any}\n", .{@typeInfo(p.child)});
//         },
//         else => {},
//     }
//     // const tca = ta.pointer.child;
//     // const tcb = tb.pointer.child;
//     // std.debug.print("type: ta= {} tb= {} tca= {} tcb= {}\n", .{ tca, tcb, tca, tcb });
// }

// test "basic add functionality" {
//     try std.testing.expect(add(3, 7) == 10);
// }
