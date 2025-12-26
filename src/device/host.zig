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
