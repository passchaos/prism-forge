const std = @import("std");
const utils = @import("utils.zig");

const host = @import("device/host.zig");

const layout_t = @import("layout.zig");
const storage_t = @import("storage.zig");
const log = @import("log.zig");

const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

var thread_pool: ?std.Thread.Pool = null;

const N_JOBS: usize = 8;

// op method
pub fn matmul(t1: anytype, t2: anytype) !Tensor(
    utils.tensor.matmulResultNDimComp(@TypeOf(t1), @TypeOf(t2)),
    .{ .T = utils.tensor.matmulResultElementTypeComp(@TypeOf(t1), @TypeOf(t2)) },
) {
    const T1 = comptime @TypeOf(t1);
    const T2 = comptime @TypeOf(t2);

    const ndim1 = comptime T1.DIMS;
    const ndim2 = comptime T2.DIMS;
    const nres = comptime utils.tensor.matmulResultNDimComp(@TypeOf(t1), @TypeOf(t2));

    const E1 = comptime T1.T;
    const E2 = comptime T2.T;
    const ER = comptime utils.tensor.matmulResultElementTypeComp(@TypeOf(t1), @TypeOf(t2));

    // only support 2D tensors for now
    if (ndim1 != 2 or ndim2 != 2) {
        @compileError("only support 2D tensor matmul");
    }

    if (E1 != E2 or (E1 != f32 and E1 != f64) or (E2 != f32 and E2 != f64)) {
        @compileError("only support f32 and f64 matmul" ++ " t1: " ++ @typeName(E1) ++ " t2: " ++ @typeName(E2));
    }

    const dt1 = @as(Tensor(ndim1, .{ .T = E1 }), t1);
    const dt2 = @as(Tensor(ndim2, .{ .T = E2 }), t2);

    const a1 = dt1.storage.allocator;
    const a2 = dt2.storage.allocator;

    if (a1.ptr != a2.ptr) {
        return error.AllocatorMismatch;
    }

    if (dt1.shape()[1] != t2.shape()[0]) {
        return error.ShapeMismatch;
    }

    // const dt1_c = try dt1.contiguous();

    const lhs = try (try dt1.contiguous()).to(ER);

    const rhs = try dt2.contiguous();

    const m = lhs.shape()[0];
    const n = rhs.shape()[1];
    const k = lhs.shape()[1];

    const a: []const ER = @ptrCast(lhs.storage.dataSlice());
    const b: [*c]const ER = @ptrCast(rhs.storage.dataSlice());

    const buf = try a1.alloc(ER, m * n);

    const c: []ER = @ptrCast(buf);

    host.matmul(ER, @as([*c]const ER, @ptrCast(a)), b, @as([*c]ER, @ptrCast(c)), m, n, k);

    if (!dt1.isContiguous()) {
        lhs.deinit();
    }

    if (!dt2.isContiguous()) {
        rhs.deinit();
    }

    const layout = layout_t.Layout(nres).init([2]usize{ m, n });
    const storage = try storage_t.Storage(ER, .Cpu).initImpl(a1, buf);

    return try tensor.Tensor(nres, .{ .T = ER }).fromDataImpl(layout, storage, 0);
}

test "matmul" {
    const allocator = std.testing.allocator;

    const t1 = try tensor.fromArray(allocator, [_][2]f64{.{ 0.6, 0.9 }});
    defer t1.deinit();
    const t2 = try tensor.fromArray(allocator, [_][3]f64{
        .{ 0.47355232, 0.9977393, 0.84668094 },
        .{ 0.85557411, 0.03563661, 0.69422093 },
    });
    defer t2.deinit();

    const t3 = try matmul(t1, t2);
    defer t3.deinit();
    const expected_t3 = try tensor.fromArray(allocator, [_][3]f64{
        .{ 1.05414809, 0.63071653, 1.1328074 },
    });
    defer expected_t3.deinit();

    const approx_compare_res = expected_t3.approxEqual(t3, 1e-8, 1e-8);
    log.print(@src(), "Approximate comparison result: {}\n", .{approx_compare_res});
    try std.testing.expect(approx_compare_res);
}
