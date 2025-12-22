const std = @import("std");

const tensor = @import("../tensor.zig");
const utils = @import("../utils.zig");

// loss
pub fn mseLoss(
    comptime N: usize,
    comptime T: type,
    lt: tensor.Tensor(N, .{ .T = T }),
    rt: tensor.Tensor(N, .{ .T = T }),
) !tensor.Tensor(0, .{ .T = T }) {
    var a = try lt.sub(rt);
    defer a.deinit();
    a.powi_(2);

    var res = try a.sumAll();
    try res.div_(2);

    return res;
}

pub fn crossEntropy(
    comptime N: usize,
    comptime T: type,
    pt: tensor.Tensor(N, .{ .T = T }),
    lt: tensor.Tensor(N, .{ .T = T }),
) !tensor.Tensor(0, .{ .T = T }) {
    switch (@typeInfo(T)) {
        .float => |_| {
            const batch_size = switch (N) {
                1 => 1,
                2 => pt.shape()[0],
                inline else => @compileError("unsuported dimension"),
            };

            var a = pt;

            const scope = struct {
                fn call(v: T, _: void) T {
                    return -@log(v + 1e-7);
                }
            };

            a.map_(void{}, scope.call);

            try a.mul_(lt);

            var res = try a.sumAll();
            try res.div_(@as(T, @floatFromInt(batch_size)));

            return res;
        },
        else => @compileError("unsupported type"),
    }
}
