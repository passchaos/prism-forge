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

pub fn svd(allocator: std.mem.Allocator, comptime T: type, a: [*c]T, m: usize, n: usize) !struct {
    u: []T,
    s: []T,
    vt: []T,
} {
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

    const res = switch (T) {
        f32 => blasapi.dgesvd_(
            job_i,
            job_i,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            a,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(s_d)),
            @as([*c]T, @ptrCast(fake_u_d)),
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(fake_vt_d)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            @as([*c]T, @ptrCast(&work_query)),
            &lwork,
            &info,
        ),
        f64 => blasapi.dgesvd_(
            job_i,
            job_i,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            a,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(s_d)),
            @as([*c]T, @ptrCast(fake_u_d)),
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(fake_vt_d)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            @as([*c]T, @ptrCast(&work_query)),
            &lwork,
            &info,
        ),
        else => @compileError("unsupported data type: " ++ @typeName(T)),
    };

    if (res != 0) return error.SvdQueryFailed;

    // std.debug.print("work_query: {any} info: {} res: {}\n", .{ work_query, info, res });

    const work_size = @as(usize, @intFromFloat(work_query[0]));
    lwork = @as(c_int, @intCast(work_size));

    var work = try allocator.alloc(T, work_size);

    const res1 = switch (T) {
        f32 => blasapi.sgesvd_(
            job_i,
            job_i,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            a,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(s_d)),
            @as([*c]T, @ptrCast(fake_u_d)),
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(fake_vt_d)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            @as([*c]T, @ptrCast(&work)),
            &lwork,
            &info,
        ),
        f64 => blasapi.dgesvd_(
            job_i,
            job_i,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            a,
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(s_d)),
            @as([*c]T, @ptrCast(fake_u_d)),
            @as([*c]c_int, @ptrCast(&fake_m)),
            @as([*c]T, @ptrCast(fake_vt_d)),
            @as([*c]c_int, @ptrCast(&fake_n)),
            @as([*c]T, @ptrCast(&work)),
            &lwork,
            &info,
        ),
        else => @compileError("unsupported data type: " ++ @typeName(T)),
    };

    if (res1 != 0) return error.SvdComputeFailed;

    // std.debug.print("work_query: info: {} res: {}\n", .{ info, res1 });
    // std.debug.print("u: {any} s: {any} vt: {any}\n", .{ fake_vt_d, s_d, fake_u_d });
    return .{
        .u = fake_vt_d,
        .s = s_d,
        .vt = fake_u_d,
    };
}
