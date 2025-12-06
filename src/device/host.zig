const builtin = @import("builtin");
const blasapi = switch (builtin.os.tag) {
    else => @cImport(@cInclude("cblas.h")),
    .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
};

pub fn matmul(a: [*]const f32, b: [*]const f32, c: [*]f32, m_i: usize, n_i: usize, k_i: usize) void {
    const m = @as(c_int, @intCast(m_i));
    const n = @as(c_int, @intCast(n_i));
    const k = @as(c_int, @intCast(k_i));
    blasapi.cblas_sgemm(blasapi.CblasRowMajor, blasapi.CblasNoTrans, blasapi.CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
}
