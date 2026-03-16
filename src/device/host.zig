const std = @import("std");
const builtin = @import("builtin");
const blasapi = switch (builtin.os.tag) {
    .linux => @cImport(@cInclude("cblas.h")),
    .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
    else => @compileError("Unsupported os"),
};

pub fn matmul(comptime T: type, a: [*c]const T, b: [*c]const T, c: [*c]T, m_i: usize, n_i: usize, k_i: usize) void {
    const m = @as(c_int, @intCast(m_i));
    const n = @as(c_int, @intCast(n_i));
    const k = @as(c_int, @intCast(k_i));

    switch (T) {
        f32 => {
            blasapi.cblas_sgemm(blasapi.CblasRowMajor, blasapi.CblasNoTrans, blasapi.CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
        },
        f64 => {
            blasapi.cblas_dgemm(blasapi.CblasRowMajor, blasapi.CblasNoTrans, blasapi.CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
        },
        else => @compileError("Unsupported type"),
    }
}

pub fn svd(allocator: std.mem.Allocator, comptime T: type, a_: []T, m: usize, n: usize) !struct {
    u: []T,
    s: []T,
    vt: []T,
} {
    const a = @as([*c]T, @ptrCast(a_));

    var fake_m = n;
    var fake_n = m;

    const fake_u_d = try allocator.alloc(T, fake_m * fake_m);
    const s_d = try allocator.alloc(T, @min(m, n));
    const fake_vt_d = try allocator.alloc(T, fake_n * fake_n);

    var job_i_r: [2]u8 = .{ 'A', 0 };
    const job_i = @as([*c]u8, @ptrCast(&job_i_r));

    var work_query: [1]T = undefined;
    var lwork: c_int = -1;
    var info: c_int = 0;

    const fake_m_c = @as([*c]c_int, @ptrCast(&fake_m));
    const fake_n_c = @as([*c]c_int, @ptrCast(&fake_n));
    const s_d_c = @as([*c]T, @ptrCast(s_d));
    const fake_u_d_c = @as([*c]T, @ptrCast(fake_u_d));
    const fake_vt_d_c = @as([*c]T, @ptrCast(fake_vt_d));

    var work_query_c = @as([*c]T, @ptrCast(&work_query));

    const svd_func = switch (T) {
        f32 => blasapi.sgesvd_,
        f64 => blasapi.dgesvd_,
        else => unreachable,
    };

    var res = svd_func(
        job_i,
        job_i,
        fake_m_c,
        fake_n_c,
        a,
        fake_m_c,
        s_d_c,
        fake_u_d_c,
        fake_m_c,
        fake_vt_d_c,
        fake_n_c,
        work_query_c,
        &lwork,
        &info,
    );

    if (res != 0) return error.SvdQueryFailed;

    const work_size = @as(usize, @intFromFloat(work_query[0]));
    lwork = @as(c_int, @intCast(work_size));

    const buf = try allocator.alloc(T, work_size);
    defer allocator.free(buf);

    work_query_c = @ptrCast(buf);

    res = svd_func(
        job_i,
        job_i,
        fake_m_c,
        fake_n_c,
        a,
        fake_m_c,
        s_d_c,
        fake_u_d_c,
        fake_m_c,
        fake_vt_d_c,
        fake_n_c,
        work_query_c,
        &lwork,
        &info,
    );

    if (res != 0) return error.SvdComputeFailed;

    return .{
        .u = fake_vt_d,
        .s = s_d,
        .vt = fake_u_d,
    };
}
