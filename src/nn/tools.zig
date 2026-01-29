const std = @import("std");
const shape_expr = @import("../shape_expr.zig");
const tensor = @import("../tensor.zig");
const utils = @import("../utils.zig");

const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;

fn computeChannelFilteredShape(
    comptime H: SizeExpr,
    comptime W: SizeExpr,
    comptime FH: SizeExpr,
    comptime FW: SizeExpr,
    comptime pads: [4]SizeExpr,
    comptime stride: SizeExpr,
) [2]SizeExpr {
    const OH = comptime H.add(pads[2]).add(pads[3]).sub(FH).div(stride).add(SizeExpr.static(1));
    const OW = comptime W.add(pads[0]).add(pads[1]).sub(FW).div(stride).add(SizeExpr.static(1));

    return [2]SizeExpr{ OH, OW };
}

fn computeColShape(
    comptime N: SizeExpr,
    comptime C: SizeExpr,
    comptime H: SizeExpr,
    comptime W: SizeExpr,
    comptime FH: SizeExpr,
    comptime FW: SizeExpr,
    comptime pads: [4]SizeExpr,
    comptime stride: SizeExpr,
) [2]SizeExpr {
    const s = computeChannelFilteredShape(H, W, FH, FW, pads, stride);
    return [2]SizeExpr{ N.mul(s[0]).mul(s[1]), C.mul(FH).mul(FW) };
}

pub fn im2col(
    comptime N: SizeExpr,
    comptime C: SizeExpr,
    comptime H: SizeExpr,
    comptime W: SizeExpr,
    comptime FH: SizeExpr,
    comptime FW: SizeExpr,
    comptime pads: [4]SizeExpr,
    comptime stride: SizeExpr,
    comptime T: type,
    allocator: std.mem.Allocator,
    input_data: *const tensor.Tensor(&.{ N, C, H, W }, T),
    shape_env: *const ShapeEnv,
) !tensor.Tensor(
    &computeColShape(N, C, H, W, FH, FW, pads, stride),
    T,
) {
    const padded_data = try input_data.pad(&pads, @as(T, 0));
    defer padded_data.deinit();

    std.debug.print("padded data: {f}\n", .{padded_data});
    const shape = padded_data.shape();

    const FILTER_OUTPUT_S = comptime computeChannelFilteredShape(H, W, FH, FW, pads, stride);
    const F_OH = FILTER_OUTPUT_S[0];
    const F_OW = FILTER_OUTPUT_S[1];

    const foh_v = try F_OH.eval(shape_env);
    const fow_v = try F_OW.eval(shape_env);

    const n_v = shape[0];
    const c_v = shape[1];

    const fh_v = try FH.eval(shape_env);
    const fw_v = try FW.eval(shape_env);

    const COL_S = comptime [2]SizeExpr{ N.mul(F_OH).mul(F_OW), C.mul(FH).mul(FW) };
    var result = try tensor.zeros(allocator, T, &COL_S, shape_env);

    for (0..n_v) |n_i| {
        for (0..c_v) |c_i| {
            for (0..foh_v) |foh_i| {
                for (0..fow_v) |fow_i| {
                    const r_idx = n_i * foh_v * fow_v + foh_i * fow_v + fow_i;

                    for (0..fh_v) |fh_i| {
                        for (0..fw_v) |fw_i| {
                            const data_idx = [4]usize{ n_i, c_i, foh_i + fh_i, fow_i + fw_i };

                            const d_v = try padded_data.getData(data_idx);

                            const c_idx = c_i * fh_v * fw_v + fh_i * fw_v + fw_i;

                            try result.setData([2]usize{ r_idx, c_idx }, d_v);
                        }
                    }
                }
            }
        }
    }

    return result;
}

pub fn col2im(
    comptime N: SizeExpr,
    comptime C: SizeExpr,
    comptime H: SizeExpr,
    comptime W: SizeExpr,
    comptime FH: SizeExpr,
    comptime FW: SizeExpr,
    comptime pads: [4]SizeExpr,
    comptime stride: SizeExpr,
    comptime T: type,
    allocator: std.mem.Allocator,
    col_data: *const tensor.Tensor(&computeColShape(N, C, H, W, FH, FW, pads, stride), T),
    shape_env: *const ShapeEnv,
) !tensor.Tensor(&.{ N, C, H, W }, T) {
    const padded_se = comptime utils.tensor.computePaddedShape(&.{ N, C, H, W }, &pads);

    var result = try tensor.zeros(
        allocator,
        T,
        &padded_se,
        shape_env,
    );
    defer result.deinit();

    const FILTER_OUTPUT_S = computeChannelFilteredShape(H, W, FH, FW, pads, stride);
    const F_OH = FILTER_OUTPUT_S[0];
    const F_OW = FILTER_OUTPUT_S[1];

    const n_v = try N.eval(shape_env);
    const c_v = try C.eval(shape_env);
    const foh_v = try F_OH.eval(shape_env);
    const fow_v = try F_OW.eval(shape_env);
    const fh_v = try FH.eval(shape_env);
    const fw_v = try FW.eval(shape_env);

    for (0..n_v) |n_i| {
        for (0..c_v) |c_i| {
            for (0..foh_v) |foh_i| {
                for (0..fow_v) |fow_i| {
                    const col_r_idx = n_i * foh_v * fow_v + foh_i * fow_v + fow_i;

                    for (0..fh_v) |fh_i| {
                        for (0..fw_v) |fw_i| {
                            const col_c_idx = c_i * fh_v * fw_v + fh_i * fw_v + fw_i;

                            const val = try col_data.getData([2]usize{ col_r_idx, col_c_idx });
                            try result.setData([4]usize{ n_i, c_i, foh_i + fh_i, fow_i + fw_i }, val);
                        }
                    }
                }
            }
        }
    }

    return try result.sliceView(&.{
        .All,
        .All,
        tensor.SliceExpr.range(pads[2], H.add(pads[2])),
        tensor.SliceExpr.range(pads[0], W.add(pads[0])),
    });
}

test "im2col" {
    const allocator = std.testing.allocator;

    const N = comptime SizeExpr.static(1);
    const C = comptime SizeExpr.static(2);
    const H = comptime SizeExpr.static(4);
    const W = comptime SizeExpr.static(4);

    const FH = comptime SizeExpr.static(3);
    const FW = comptime SizeExpr.static(3);
    const PAD = comptime SizeExpr.static(1);
    const STRIDE = comptime SizeExpr.static(1);

    const pads = [4]SizeExpr{ PAD, PAD, PAD, PAD };

    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    var raw_data = try tensor.fromArray(allocator, [2][4][4]f32{ [4][4]f32{
        [4]f32{ 1.0, 2.0, 3.0, 0.0 },
        [4]f32{ 0.0, 1.0, 2.0, 3.0 },
        [4]f32{ 3.0, 0.0, 1.0, 2.0 },
        [4]f32{ 2.0, 3.0, 0.0, 1.0 },
    }, [4][4]f32{
        [4]f32{ 3.0, 6.0, 9.0, 0.0 },
        [4]f32{ 0.0, 3.0, 6.0, 3.0 },
        [4]f32{ 6.0, 0.0, 3.0, 1.0 },
        [4]f32{ 5.0, 9.0, 0.0, 2.0 },
    } }, &shape_env);
    defer raw_data.deinit();

    const input_data = try raw_data.reshape(&.{ N, C, H, W });
    defer input_data.deinit();

    const col_data = try im2col(N, C, H, W, FH, FW, pads, STRIDE, f32, allocator, &input_data, &shape_env);
    defer col_data.deinit();

    const expected_col_data: [16][18]f32 = .{
        .{ 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 3, 6, 0, 0, 3 },
        .{ 0, 0, 0, 1, 2, 3, 0, 1, 2, 0, 0, 0, 3, 6, 9, 0, 3, 6 },
        .{ 0, 0, 0, 2, 3, 0, 1, 2, 3, 0, 0, 0, 6, 9, 0, 3, 6, 3 },
        .{ 0, 0, 0, 3, 0, 0, 2, 3, 0, 0, 0, 0, 9, 0, 0, 6, 3, 0 },
        .{ 0, 1, 2, 0, 0, 1, 0, 3, 0, 0, 3, 6, 0, 0, 3, 0, 6, 0 },
        .{ 1, 2, 3, 0, 1, 2, 3, 0, 1, 3, 6, 9, 0, 3, 6, 6, 0, 3 },
        .{ 2, 3, 0, 1, 2, 3, 0, 1, 2, 6, 9, 0, 3, 6, 3, 0, 3, 1 },
        .{ 3, 0, 0, 2, 3, 0, 1, 2, 0, 9, 0, 0, 6, 3, 0, 3, 1, 0 },
        .{ 0, 0, 1, 0, 3, 0, 0, 2, 3, 0, 0, 3, 0, 6, 0, 0, 5, 9 },
        .{ 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 3, 6, 6, 0, 3, 5, 9, 0 },
        .{ 1, 2, 3, 0, 1, 2, 3, 0, 1, 3, 6, 3, 0, 3, 1, 9, 0, 2 },
        .{ 2, 3, 0, 1, 2, 0, 0, 1, 0, 6, 3, 0, 3, 1, 0, 0, 2, 0 },
        .{ 0, 3, 0, 0, 2, 3, 0, 0, 0, 0, 6, 0, 0, 5, 9, 0, 0, 0 },
        .{ 3, 0, 1, 2, 3, 0, 0, 0, 0, 6, 0, 3, 5, 9, 0, 0, 0, 0 },
        .{ 0, 1, 2, 3, 0, 1, 0, 0, 0, 0, 3, 1, 9, 0, 2, 0, 0, 0 },
        .{ 1, 2, 0, 0, 1, 0, 0, 0, 0, 3, 1, 0, 0, 2, 0, 0, 0, 0 },
    };
    const expected_col_t = try tensor.fromArray(allocator, expected_col_data, &shape_env);
    defer expected_col_t.deinit();

    const expected_col_equal_res = col_data.equal(&expected_col_t);
    try std.testing.expect(expected_col_equal_res);

    const orig_data = try col2im(N, C, H, W, FH, FW, pads, STRIDE, f32, allocator, &col_data, &shape_env);
    defer orig_data.deinit();

    const equal_res = input_data.equal(&orig_data);
    try std.testing.expect(equal_res);

    std.debug.print("data: \tinput= {f}\n\tcol= {f}\n\torig= {f}\n", .{ input_data, col_data, orig_data });
    std.debug.print("equal result: {}\n", .{equal_res});
}
