const std = @import("std");
pub fn print(comptime src: std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
    std.debug.print("{s} {s}:{d} {s} " ++ format, .{ src.module, src.file, src.line, src.fn_name } ++ args);
}
