const std = @import("std");
const utils = @import("./utils.zig");
const product = utils.product;

pub fn Layout(comptime N: usize) type {
    return struct {
        _shape: [N]usize,
        _stride: [N]usize,
        _is_contiguous: bool = true,

        const Self = @This();

        pub fn broadcastTo(self: Self, comptime BN: usize, target_shape: [BN]usize) !Layout(BN) {
            if (std.mem.eql(usize, &self.shape(), &target_shape)) {
                return self;
            }

            const new_shape = try utils.broadcastShapes(N, BN, self.shape(), target_shape);
            return Layout(BN).init(new_shape);
        }

        pub fn cat(layouts: []const Self, dim: usize) !Self {
            if (layouts.len == 0) return error.EmptyShapes;
            var base_shape = layouts[0].shape();

            if (layouts.len == 1) {
                return Self.init(base_shape);
            }

            for (layouts[1..]) |shape_i| {
                for (shape_i.shape(), 0..) |dim_size, i| {
                    if (i == dim) {
                        base_shape[i] += dim_size;
                    } else {
                        if (dim_size != base_shape[i]) {
                            return error.IncompatibleShapes;
                        }
                    }
                }
            }

            return Self.init(base_shape);
        }

        pub fn stack(layouts: []const Self, dim: usize) !Layout(N + 1) {
            if (layouts.len == 0) return error.EmptyShapes;

            {
                const base_shape = layouts[0].shape();
                for (layouts) |shape_i| {
                    if (!std.mem.eql(usize, &base_shape, &shape_i.shape())) {
                        return error.ShapeMustBeEqual;
                    }
                }
            }

            const count = layouts.len;
            var base_shape = [_]usize{0} ** (N + 1);

            var i: usize = 0;
            var j: usize = 0;
            while (i < N + 1) {
                if (i == dim) {
                    base_shape[i] = count;
                    i += 1;
                }
                base_shape[i] = layouts[0].shape()[j];

                i += 1;
                j += 1;
            }

            return Layout(N + 1).init(base_shape);
        }

        fn checkContiguous(shapes_a: []const usize, strides_a: []const usize) bool {
            var expected_stride: usize = 1;
            var i: usize = shapes_a.len;

            while (i > 0) : (i -= 1) {
                const dim = shapes_a[i - 1];
                const stride_val = strides_a[i - 1];

                if (stride_val != expected_stride) {
                    return false;
                } else {
                    expected_stride *= dim;
                }
            }

            return true;
        }

        pub fn init(shapes_a: [N]usize) Self {
            const strides_i = utils.computeArrayShapeStrides(N, shapes_a);

            return Self.initRaw(shapes_a, strides_i);
        }

        pub fn initRaw(shapes_a: [N]usize, strides_a: [N]usize) Self {
            const is_contiguous = checkContiguous(&shapes_a, &strides_a);

            const layout = Self{
                ._shape = shapes_a,
                ._stride = strides_a,
                ._is_contiguous = is_contiguous,
            };

            return layout;
        }

        pub fn transpose(self: *const Self, dim0: usize, dim1: usize) !Self {
            if (dim0 >= N or dim1 >= N) return error.InvalidDim;

            var perm = [_]usize{0} ** N;

            for (&perm, 0..) |*p, i| {
                if (i == dim0) {
                    p.* = dim1;
                } else if (i == dim1) {
                    p.* = dim0;
                } else {
                    p.* = i;
                }
            }

            return try self.permute(perm);
        }

        pub fn permute(self: *const Self, perm: [N]usize) !Self {
            var new_shapes = self._shape;
            var new_strides = self._stride;

            for (perm, 0..) |p, i| {
                if (p >= N) return error.InvalidDim;

                new_shapes[i] = self._shape[p];
                new_strides[i] = self._stride[p];
            }

            return Self.initRaw(new_shapes, new_strides);
        }

        pub fn reshape(self: *const Self, new_shapes: anytype) !Layout(utils.array.getArrayShapeComp(@TypeOf(new_shapes))[0]) {
            const N1 = comptime utils.array.getArrayShapeComp(@TypeOf(new_shapes))[0];
            const new_size = product(&new_shapes);

            if (new_size != self.size()) {
                return error.InvalidShape;
            }

            return Layout(N1).init(new_shapes);
        }

        pub fn unsqueeze(self: *const Self, dim: usize) !Layout(N + 1) {
            if (dim > N) return error.InvalidDim;

            var new_shapes = [_]usize{0} ** (N + 1);

            var i: usize = 0;
            var j: usize = 0;

            while (i < N + 1) {
                if (i == dim) {
                    new_shapes[i] = 1;
                    i += 1;
                } else {
                    new_shapes[i] = self._shape[j];
                    i += 1;
                    j += 1;
                }
            }

            return Layout(N + 1).init(new_shapes);
        }

        pub fn squeeze(self: *const Self, dim: usize) !Layout(N - 1) {
            if (dim > N) return error.InvalidDim;
            if (self.shape()[dim] != 1) return error.DimNotOne;

            var new_shapes = [_]usize{0} ** (N - 1);

            var i: usize = 0;
            var j: usize = 0;

            while (i < N - 1) {
                if (j == dim) {
                    j += 1;
                } else {
                    new_shapes[i] = self._shape[j];
                    i += 1;
                    j += 1;
                }
            }

            return Layout(N - 1).init(new_shapes);
        }

        pub fn iter(self: *const Self) ShapeIterator(N) {
            return ShapeIterator(N).init(self.shape());
        }

        pub fn clone(self: *const Self) Self {
            return Self{
                ._shape = self._shape,
                ._stride = self._stride,
            };
        }

        pub fn equal(self: *const Self, other: *const Self) bool {
            return self._shape == other._shape and self._stride == other._stride;
        }

        pub fn size(self: *const Self) usize {
            return product(&self._shape);
        }

        pub fn ndim(self: *const Self) usize {
            return self._shape.len;
        }

        pub fn shape(self: *const Self) [N]usize {
            return self._shape;
        }

        pub fn stride(self: *const Self) [N]usize {
            return self._stride;
        }

        pub fn isContiguous(self: *const Self) bool {
            return self._is_contiguous;
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print("Layout {{\n", .{});
            try writer.print("  shapes: {any},\n", .{self._shape});
            try writer.print("  strides: {any},\n", .{self._stride});
            try writer.print("  contiguous: {}\n", .{self._is_contiguous});
            try writer.print("}}", .{});
        }
    };
}

test "transpose" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });
    const transposed = try layout.transpose(0, 2);
    try std.testing.expectEqual(transposed._shape, [_]usize{ 4, 3, 2 });
    try std.testing.expectEqual(transposed._stride, [_]usize{ 1, 4, 12 });
}

test "permute" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });
    const permuted = try layout.permute([_]usize{ 2, 0, 1 });
    try std.testing.expectEqual(permuted._shape, [_]usize{ 4, 2, 3 });
    try std.testing.expectEqual(permuted._stride, [_]usize{ 1, 12, 4 });
}

test "reshape" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });

    try std.testing.expectError(error.InvalidShape, layout.reshape([_]usize{ 2, 6 }));
    const reshaped = try layout.reshape([_]usize{ 6, 4 });
    try std.testing.expectEqual(reshaped._shape, [_]usize{ 6, 4 });
    try std.testing.expectEqual(reshaped._stride, [_]usize{ 4, 1 });
}

test "unsqueeze" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });

    std.debug.print("layout: {f}\n", .{layout});
    const unsqueezed = try layout.unsqueeze(1);
    try std.testing.expectEqual(unsqueezed._shape, [_]usize{ 2, 1, 3, 4 });
    try std.testing.expectEqual(unsqueezed._stride, [_]usize{ 12, 12, 4, 1 });

    const squeezed = try unsqueezed.squeeze(1);
    try std.testing.expectEqual(squeezed._shape, [_]usize{ 2, 3, 4 });
    try std.testing.expectEqual(squeezed._stride, [_]usize{ 12, 4, 1 });
    std.debug.print("squeezed: {f}\n", .{squeezed});
}

test "cat or stack" {
    const Layout3 = Layout(3);

    const l1 = Layout3.init([_]usize{ 2, 3, 4 });
    const l2 = Layout3.init([_]usize{ 1, 3, 4 });
    const l3 = Layout3.init([_]usize{ 2, 3, 4 });

    const lc = try Layout3.cat(&.{ l1, l2, l3 }, 0);

    try std.testing.expectError(error.IncompatibleShapes, Layout3.cat(&.{ l1, l2, l3 }, 2));

    try std.testing.expectError(error.ShapeMustBeEqual, Layout3.stack(&.{ l1, l2, l3 }, 2));

    const ls = try Layout3.stack(&.{ l1, l3 }, 2);
    try std.testing.expectEqualDeep(ls.shape(), [4]usize{ 2, 3, 2, 4 });
    std.debug.print("lc: {f} ls: {f}\n", .{ lc, ls });

    // std.debug.print("layout: {f}\n", .{layout});
    // const stacked = try layout.stack(1, layout);
    // try std.testing.expectEqual(stacked._shape, [_]usize{ 2, 2, 3, 4 });
    // try std.testing.expectEqual(stacked._stride, [_]usize{ 12, 12, 4, 1 });

    // const cat = try layout.cat(1, layout);
    // try std.testing.expectEqual(cat._shape, [_]usize{ 2, 6, 4 });
    // try std.testing.expectEqual(cat._stride, [_]usize{ 12, 4, 1 });
}

pub fn ShapeIterator(comptime N: usize) type {
    return struct {
        _shapes: [N]usize,

        idx: [N]usize,
        done: bool,

        pub fn init(shapes_a: [N]usize) @This() {
            const idx = [_]usize{0} ** N;

            return @This(){
                ._shapes = shapes_a,
                .idx = idx,
                .done = false,
            };
        }

        pub fn next(self: *@This()) ?[N]usize {
            if (self.done) return null;

            const outer_indices = self.idx;

            var d: usize = self._shapes.len;

            // handle zero-demension
            if (d == 0) {
                self.done = true;
                return self.idx;
            }

            while (d > 0) : (d -= 1) {
                self.idx[d - 1] += 1;

                if (self.idx[d - 1] < self._shapes[d - 1]) {
                    break;
                }
                self.idx[d - 1] = 0;

                if (d == 1) self.done = true;
            }

            return outer_indices;
        }
    };
}

test "shape iter" {
    const Layout3 = Layout(3);
    const layout = Layout3.init([_]usize{ 2, 3, 4 });

    var iter = layout.iter();

    while (iter.next()) |idx| {
        std.debug.print("idx: {any}\n", .{idx});
    }
}
