const builtin = @import("builtin");
const c = switch (builtin.os.tag) {
    .linux => @cImport(@cInclude("cblas.h")),
    .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
};

pub fn matmul(a: *const f32, b: *const f32, c: *f32, m: usize, n: usize, k: usize) void {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
}
