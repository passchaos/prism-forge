const std = @import("std");
const tensor = @import("../tensor.zig");

pub fn numericalGradient(
    allocator: std.mem.Allocator,
    comptime N: usize,
    f: fn (comptime N: usize, tensor.Tensor(N, .{})) f32,
    tval: tensor.Tensor(N, .{}),
) !tensor.Tensor(N, .{}) {
    const h = 1e-4;

    var tval_v = tval;
    var grad = try tensor.zerosLike(allocator, tval);

    var x_v_iter = tval_v.shapeIter();
    while (x_v_iter.next()) |idx| {
        const tmp_val = try tval_v.getData(idx);

        try tval_v.setData(idx, tmp_val + h);
        const fxh1 = f(N, tval_v);

        try tval_v.setData(idx, tmp_val - h);
        const fxh2 = f(N, tval_v);

        try grad.setData(idx, (fxh1 - fxh2) / (h + h));
    }

    return grad;
}

pub fn gradientDescent(
    allocator: std.mem.Allocator,
    comptime N: usize,
    f: fn (
        comptime N: usize,
        tensor.Tensor(N, .{}),
    ) f32,
    init_x: *tensor.Tensor(N, .{}),
    args: struct { lr: f32 = 0.01, step_number: usize = 100 },
) !void {
    for (0..args.step_number) |_| {
        var grad = try numericalGradient(allocator, N, f, init_x.*);
        defer grad.deinit();

        try grad.mul_(args.lr);

        try init_x.sub_(grad);
    }
}
