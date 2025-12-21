//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const tensor = @import("tensor.zig");
const basic = @import("nn/basic.zig");
const matmul = @import("matmul.zig");

// pub fn logFn(
//     comptime level: std.log.Level,
//     comptime scope: @TypeOf(.enum_literal),
//     comptime format: []const u8,
//     args: anytype,
// ) void {
//     _ = scope;
//     const src = @src();
//     std.debug.print("{s}:{d} [{s}] " ++ format ++ "\n", .{
//         src.file, src.line, @tagName(level),
//     } ++ args);
//     // std.log.defaultLog(level, scope, format, args);
// }

// pub const std_options: std.Options = .{
//     .log_level = .debug,
//     .logFn = logFn,
// };

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
