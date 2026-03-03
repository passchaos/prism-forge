const std = @import("std");
const tensor = @import("../tensor.zig");
const shape_expr = @import("../shape_expr.zig");

const Tensor = tensor.Tensor;
const fromArray = tensor.fromArray;
const ShapeEnv = shape_expr.ShapeEnv;
const SizeExpr = shape_expr.SizeExpr;

const EPSILON: f64 = 1e-10;
const MAX_ITERATIONS: usize = 100;

fn svdResult(
    comptime m_se: SizeExpr,
    comptime n_se: SizeExpr,
    comptime T: type,
) type {
    return struct {
        u: Tensor(&.{ m_se, m_se }, T),
        s: Tensor(&.{ m_se, n_se }, T),
        vt: Tensor(&.{ n_se, n_se }, T),
    };
}

fn maxAbsOffDiagonal(mat: []const f64, n: usize) struct { row: usize, col: usize, value: f64 } {
    var max_val: f64 = 0;
    var max_row: usize = 0;
    var max_col: usize = 0;

    for (0..n) |i| {
        for (i + 1..n) |j| {
            const idx = i * n + j;
            const abs_val = @abs(mat[idx]);
            if (abs_val > max_val) {
                max_val = abs_val;
                max_row = i;
                max_col = j;
            }
        }
    }

    return .{ .row = max_row, .col = max_col, .value = max_val };
}

fn jacobiRotation(a_ij: f64, a_ii: f64, a_jj: f64) struct { c: f64, s: f64, tau: f64 } {
    if (@abs(a_ij) < EPSILON) {
        return .{ .c = 1, .s = 0, .tau = 0 };
    }

    const diff = a_jj - a_ii;
    const sign: f64 = if (diff >= 0) 1.0 else -1.0;
    const t = sign * a_ij / (@abs(diff) + @sqrt(a_ij * a_ij + diff * diff));
    const c = 1 / @sqrt(1 + t * t);
    const s = t * c;
    const tau = s / (1 + c);

    return .{ .c = c, .s = s, .tau = tau };
}

pub fn svd(
    allocator: std.mem.Allocator,
    comptime m_se: SizeExpr,
    comptime n_se: SizeExpr,
    comptime T: type,
    shape_env: *const ShapeEnv,
    a: *const Tensor(&.{ m_se, n_se }, T),
) !svdResult(m_se, n_se, T) {
    const shape = a.shape();
    const m = shape[0];
    const n = shape[1];

    const min_dim = @min(m, n);

    var a_copy = try a.clone();
    defer a_copy.deinit();

    const a_data = a_copy.storage.dataSlice();

    var u_data = try allocator.alloc(f64, m * m);
    @memset(u_data, 0);
    for (0..m) |i| {
        u_data[i * m + i] = 1;
    }

    var vt_data = try allocator.alloc(f64, n * n);
    @memset(vt_data, 0);
    for (0..n) |i| {
        vt_data[i * n + i] = 1;
    }

    var iter: usize = 0;
    while (iter < MAX_ITERATIONS) : (iter += 1) {
        const a_slice = a_data[0..(min_dim * min_dim)];
        const max_off = maxAbsOffDiagonal(a_slice, min_dim);

        if (max_off.value < EPSILON) {
            break;
        }

        const row = max_off.row;
        const col = max_off.col;

        const a_ii = a_data[row * min_dim + row];
        const a_ij = a_data[row * min_dim + col];
        const a_jj = a_data[col * min_dim + col];

        const rot = jacobiRotation(a_ij, a_ii, a_jj);

        const temp_a = try allocator.alloc(f64, min_dim * min_dim);
        @memcpy(temp_a, a_data[0..(min_dim * min_dim)]);
        defer allocator.free(temp_a);

        for (0..min_dim) |i| {
            if (i != row and i != col) {
                const g = temp_a[row * min_dim + i];
                const h = temp_a[col * min_dim + i];
                a_data[row * min_dim + i] = g - rot.s * (h + g * rot.tau);
                a_data[i * min_dim + row] = a_data[row * min_dim + i];
                a_data[col * min_dim + i] = h + rot.s * (g - h * rot.tau);
                a_data[i * min_dim + col] = a_data[col * min_dim + i];
            }
        }

        const g = temp_a[row * min_dim + row];
        const h = temp_a[col * min_dim + col];
        a_data[row * min_dim + row] = g - rot.s * (h + g * rot.tau);
        a_data[col * min_dim + col] = h + rot.s * (g - h * rot.tau);
        a_data[row * min_dim + col] = 0;
        a_data[col * min_dim + row] = 0;

        const temp_u = try allocator.alloc(f64, m * m);
        @memcpy(temp_u, u_data);
        defer allocator.free(temp_u);

        for (0..m) |i| {
            const g_u = temp_u[i * m + row];
            const h_u = temp_u[i * m + col];
            u_data[i * m + row] = g_u - rot.s * (h_u + g_u * rot.tau);
            u_data[i * m + col] = h_u + rot.s * (g_u - h_u * rot.tau);
        }

        const temp_vt = try allocator.alloc(f64, n * n);
        @memcpy(temp_vt, vt_data);
        defer allocator.free(temp_vt);

        for (0..n) |i| {
            const g_v = temp_vt[row * n + i];
            const h_v = temp_vt[col * n + i];
            vt_data[row * n + i] = g_v - rot.s * (h_v + g_v * rot.tau);
            vt_data[col * n + i] = h_v + rot.s * (g_v - h_v * rot.tau);
        }
    }

    var s_values = try allocator.alloc(f64, min_dim);
    for (0..min_dim) |i| {
        s_values[i] = @abs(a_data[i * min_dim + i]);
    }

    var sort_indices = try allocator.alloc(usize, min_dim);
    for (0..min_dim) |i| {
        sort_indices[i] = i;
    }

    for (0..min_dim) |i| {
        for (i + 1..min_dim) |j| {
            if (s_values[j] > s_values[i]) {
                std.mem.swap(f64, &s_values[i], &s_values[j]);
                std.mem.swap(usize, &sort_indices[i], &sort_indices[j]);
            }
        }
    }

    var sorted_u = try allocator.alloc(f64, m * m);
    for (0..m) |i| {
        const src = u_data[i * m .. i * m + m];
        @memcpy(sorted_u[i * m .. i * m + m], src);
    }
    for (0..m) |i| {
        const old_idx = sort_indices[i];
        if (old_idx != i) {
            for (0..m) |j| {
                sorted_u[j * m + i] = u_data[j * m + old_idx];
            }
        }
    }

    var sorted_vt = try allocator.alloc(f64, n * n);
    for (0..n) |i| {
        const src = vt_data[i * n .. i * n + n];
        @memcpy(sorted_vt[i * n .. i * n + n], src);
    }
    for (0..n) |i| {
        const old_idx = sort_indices[i];
        if (old_idx != i) {
            for (0..n) |j| {
                sorted_vt[i * n + j] = vt_data[old_idx * n + j];
            }
        }
    }

    const u_tensor = try tensor.fromData(
        f64,
        allocator,
        sorted_u,
        &.{ m_se, m_se },
        shape_env,
    );
    allocator.free(sorted_u);

    const s_tensor = try tensor.fromData(
        f64,
        allocator,
        s_values,
        &.{ m_se, n_se },
        shape_env,
    );
    allocator.free(s_values);

    const vt_tensor = try tensor.fromData(
        f64,
        allocator,
        sorted_vt,
        &.{ n_se, n_se },
        shape_env,
    );
    allocator.free(sorted_vt);

    allocator.free(u_data);
    allocator.free(vt_data);

    return svdResult(m_se, n_se, T){
        .u = u_tensor,
        .s = s_tensor,
        .vt = vt_tensor,
    };
}

pub fn thinSvd(
    allocator: std.mem.Allocator,
    comptime m_se: SizeExpr,
    comptime n_se: SizeExpr,
    comptime T: type,
    a: *const Tensor(&.{ m_se, n_se }, T),
) !svdResult(m_se, n_se, T) {
    const shape = a.shape();
    const m = shape[0];
    const n = shape[1];

    if (m >= n) {
        const result = try svd(allocator, a);
        const vt = try result.vt.reshape(&.{ n, n });
        result.vt.deinit();
        return svdResult(m_se, n_se, T){
            .u = result.u,
            .s = result.s,
            .vt = vt,
        };
    } else {
        const a_t = try a.transpose();
        defer a_t.deinit();

        const result = try svd(allocator, a_t);

        const u = try result.vt.transpose();
        result.vt.deinit();

        const s = result.s;
        const vt = try result.u.transpose();
        result.u.deinit();

        return svdResult(m_se, n_se, T){
            .u = u,
            .s = s,
            .vt = vt,
        };
    }
}

test "svd" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    var shape_env = try ShapeEnv.init(allocator);
    defer shape_env.deinit();

    const matrix_data = [2][3]f64{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    };

    const a = try fromArray(allocator, matrix_data, &shape_env);
    const result = try svd(
        allocator,
        SizeExpr.static(2),
        SizeExpr.static(3),
        f64,
        &shape_env,
        &a,
    );

    std.debug.print("U: {any}\n", .{result.u});
    std.debug.print("S: {any}\n", .{result.s});
    std.debug.print("VT: {any}\n", .{result.vt});

    result.u.deinit();
    result.s.deinit();
    result.vt.deinit();
}
