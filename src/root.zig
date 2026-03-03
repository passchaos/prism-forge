//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const shape_expr = @import("shape_expr.zig");
pub const nn = @import("nn/mod.zig");
pub const plot = @import("plot.zig");
pub const log = @import("log.zig");
pub const mnist = @import("mnist.zig");
pub const matrix = @import("matrix/mod.zig");

pub const optim = nn.optim;
pub const mlp = nn.model.mlp;
pub const conv_net = nn.model.conv_net;

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

test {
    std.testing.refAllDecls(@This());
}
