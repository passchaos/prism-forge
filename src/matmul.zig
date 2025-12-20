const std = @import("std");
const utils = @import("utils.zig");

const host = @import("device/host.zig");

const layout_t = @import("layout.zig");
const storage_t = @import("storage.zig");

const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

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

    const lhs = try (try dt1.contiguous()).to(ER);
    const rhs = try (try dt2.contiguous()).to(ER);

    const m = lhs.shape()[0];
    const n = rhs.shape()[1];
    const k = lhs.shape()[1];

    const a: [*c]const ER = @ptrCast(lhs.storage.dataSlice());
    const b: [*c]const ER = @ptrCast(rhs.storage.dataSlice());

    const buf = try a1.alloc(ER, m * n);

    const c: [*c]ER = @ptrCast(buf);

    host.matmul(ER, a, b, c, m, n, k);

    const layout = layout_t.Layout(nres).init([2]usize{ m, n });
    const storage = try storage_t.Storage(ER, .Cpu).initImpl(a1, buf);

    return try tensor.Tensor(nres, .{ .T = ER }).fromDataImpl(layout, storage, 0);
}
