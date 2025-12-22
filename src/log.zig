const std = @import("std");
pub fn print(comptime src: std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
    std.debug.print("{s}:{d} " ++ format, .{ src.file, src.line } ++ args);
}
