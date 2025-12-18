const std = @import("std");
const utils = @import("./utils.zig");
const product = utils.product;

pub fn Layout(comptime N: usize) type {
    return struct {
        _shapes: [N]usize,
        _strides: [N]usize,
        _is_contiguous: bool = true,

        const Self = @This();

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
                ._shapes = shapes_a,
                ._strides = strides_a,
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
            var new_shapes = self._shapes;
            var new_strides = self._strides;

            for (perm, 0..) |p, i| {
                if (p >= N) return error.InvalidDim;

                new_shapes[i] = self._shapes[p];
                new_strides[i] = self._strides[p];
            }

            return Self.initRaw(new_shapes, new_strides);
        }

        pub fn reshape(self: *const Self, new_shapes: anytype) !Layout(utils.getCompArrayLen(@TypeOf(new_shapes))) {
            const N1 = comptime utils.getCompArrayLen(@TypeOf(new_shapes));
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
                    new_shapes[i] = self._shapes[j];
                    i += 1;
                    j += 1;
                }
            }

            return Layout(N + 1).init(new_shapes);
        }

        pub fn squeeze(self: *const Self, dim: usize) !Layout(N - 1) {
            if (dim > N) return error.InvalidDim;
            if (self.shapes()[dim] != 1) return error.DimNotOne;

            var new_shapes = [_]usize{0} ** (N - 1);

            var i: usize = 0;
            var j: usize = 0;

            while (i < N - 1) {
                if (j == dim) {
                    j += 1;
                } else {
                    new_shapes[i] = self._shapes[j];
                    i += 1;
                    j += 1;
                }
            }

            return Layout(N - 1).init(new_shapes);
        }

        pub fn iter(self: *const Self) ShapeIterator(N) {
            return ShapeIterator(N).init(self.shapes());
        }

        pub fn clone(self: *const Self) Self {
            return Self{
                ._shapes = self._shapes,
                ._strides = self._strides,
            };
        }

        pub fn equal(self: *const Self, other: *const Self) bool {
            return self._shapes == other._shapes and self._strides == other._strides;
        }

        pub fn size(self: *const Self) usize {
            return product(&self._shapes);
        }

        pub fn ndim(self: *const Self) usize {
            return self._shapes.len;
        }

        pub fn shapes(self: *const Self) [N]usize {
            return self._shapes;
        }

        pub fn strides(self: *const Self) [N]usize {
            return self._strides;
        }

        pub fn isContiguous(self: *const Self) bool {
            return self._is_contiguous;
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print("Layout {{\n", .{});
            try writer.print("  shapes: {any},\n", .{self._shapes});
            try writer.print("  strides: {any},\n", .{self._strides});
            try writer.print("  contiguous: {}\n", .{self._is_contiguous});
            try writer.print("}}\n", .{});
        }
    };
}

test "transpose" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });
    const transposed = try layout.transpose(0, 2);
    try std.testing.expectEqual(transposed._shapes, [_]usize{ 4, 3, 2 });
    try std.testing.expectEqual(transposed._strides, [_]usize{ 1, 4, 12 });
}

test "permute" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });
    const permuted = try layout.permute([_]usize{ 2, 0, 1 });
    try std.testing.expectEqual(permuted._shapes, [_]usize{ 4, 2, 3 });
    try std.testing.expectEqual(permuted._strides, [_]usize{ 1, 12, 4 });
}

test "reshape" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });

    try std.testing.expectError(error.InvalidShape, layout.reshape([_]usize{ 2, 6 }));
    const reshaped = try layout.reshape([_]usize{ 6, 4 });
    try std.testing.expectEqual(reshaped._shapes, [_]usize{ 6, 4 });
    try std.testing.expectEqual(reshaped._strides, [_]usize{ 4, 1 });
}

test "unsqueeze" {
    const Layout3 = Layout(3);
    const layout = Layout3.initRaw([_]usize{ 2, 3, 4 }, [_]usize{ 12, 4, 1 });

    std.debug.print("layout: {f}\n", .{layout});
    const unsqueezed = try layout.unsqueeze(1);
    try std.testing.expectEqual(unsqueezed._shapes, [_]usize{ 2, 1, 3, 4 });
    try std.testing.expectEqual(unsqueezed._strides, [_]usize{ 12, 12, 4, 1 });

    const squeezed = try unsqueezed.squeeze(1);
    try std.testing.expectEqual(squeezed._shapes, [_]usize{ 2, 3, 4 });
    try std.testing.expectEqual(squeezed._strides, [_]usize{ 12, 4, 1 });
    std.debug.print("squeezed: {f}\n", .{squeezed});
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
