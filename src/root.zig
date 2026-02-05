//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const shape_expr = @import("shape_expr.zig");
pub const optim = @import("nn/optim.zig");
pub const conv_net = @import("nn/model/conv_net.zig");
pub const mlp = @import("nn/model/mlp.zig");
pub const plot = @import("plot.zig");
pub const log = @import("log.zig");
pub const mnist = @import("mnist.zig");

pub fn logFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = scope;

    const src = @src();
    log.print(@src(), "{s}:{d} [{s}] " ++ format ++ "\n", .{
        src.file, src.line, @tagName(level),
    } ++ args);
    // std.log.defaultLog(level, scope, format, args);
}

pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = logFn,
};
