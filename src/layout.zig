const std = @import("std");
const utils = @import("./utils.zig");
const log = @import("log.zig");
const product = utils.product;

pub fn Layout(comptime SI: []const usize) type {
    return struct {
        const S = SI;
        const N = SI.len;
        const IS_LAYOUT = true;

        _stride: [N]usize,
        _is_contiguous: bool = true,

        const Self = @This();

        pub fn broadcastTo(self: Self, comptime BN: usize, target_shape: [BN]usize) !Layout(BN) {
            if (std.mem.eql(usize, &self.shape(), &target_shape)) {
                return self;
            }

            const new_stride = try utils.generateBroadcastStride(N, BN, self.shape(), self.stride(), target_shape);
            return Layout(BN).initRaw(target_shape, new_stride);
        }

        fn checkContiguous(strides_a: []const usize) bool {
            var expected_stride: usize = 1;
            var i: usize = SI.len;

            while (i > 0) : (i -= 1) {
                if (SI.len == 0) {
                    return true;
                } else {
                    const dim = SI[i - 1];
                    const stride_val = strides_a[i - 1];

                    if (stride_val != expected_stride) {
                        return false;
                    } else {
                        expected_stride *= dim;
                    }
                }
            }

            return true;
        }

        pub fn init() Self {
            const strides_i = computeSliceShapeStrides(N, SI);

            return Self.initRaw(strides_i);
        }

        pub fn initRaw(strides_a: [N]usize) Self {
            const is_contiguous = checkContiguous(&strides_a);

            const layout = Self{
                ._stride = strides_a,
                ._is_contiguous = is_contiguous,
            };

            return layout;
        }

        pub fn transpose(self: *const Self, comptime dim0: usize, comptime dim1: usize) Layout(&computePermutedShape(
            N,
            SI,
            computeTransposedPerm(N, dim0, dim1),
        )) {
            if (dim0 >= N or dim1 >= N) @compileError("Invalid dimension");

            const perm = comptime computeTransposedPerm(N, dim0, dim1);

            return self.permute(perm);
        }

        pub fn permute(self: *const Self, comptime perm: [N]usize) Layout(
            &computePermutedShape(
                SI,
                &perm,
            ),
        ) {
            const new_shape = comptime computePermutedShape(SI, &perm);

            var new_strides = self._stride;

            inline for (perm, 0..) |p, i| {
                if (p >= N) @compileError("Invalid dimension");

                new_strides[i] = self._stride[p];
            }

            return Layout(&new_shape).initRaw(new_strides);
        }

        pub fn reshape(_: *const Self, comptime new_shapes: []const usize) Layout(new_shapes) {
            const self_size = comptime if (N == 0) 1 else product(SI);
            const new_size = comptime product(new_shapes);

            if (comptime new_size != self_size) @compileError("Invalid shape");

            return Layout(new_shapes).init();
        }

        pub fn unsqueeze(_: *const Self, comptime dim: usize) Layout(&utils.array.insertDimComptime(
            N,
            SI,
            dim,
            1,
        )) {
            const new_shape = comptime utils.array.insertDimComptime(N, SI, dim, 1);

            return Layout(&new_shape).init();
        }

        pub fn squeeze(_: *const Self, comptime dim: usize) Layout(&utils.array.removeDimComptime(N, SI, dim)) {
            var new_shapes = [_]usize{0} ** (N - 1);

            var i: usize = 0;
            var j: usize = 0;

            while (i < N - 1) {
                if (j == dim) {
                    j += 1;
                } else {
                    new_shapes[i] = SI[j];
                    i += 1;
                    j += 1;
                }
            }

            return Layout(&utils.array.removeDimComptime(N, SI, dim)).init();
        }

        pub fn iter() ShapeIterator(SI) {
            return ShapeIterator(SI).init();
        }

        pub fn equal(self: Self, other: Self) bool {
            return (self.shape() == other.shape()) and (self.stride() == other.stride());
        }

        pub fn size(_: *const Self) usize {
            if (N == 0) return 1;
            return comptime product(SI);
        }

        pub fn ndim(_: *const Self) usize {
            return N;
        }

        pub fn shape(_: *const Self) [N]usize {
            return utils.array.comptimeSliceToArray(SI);
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
            try writer.print("Layout {{", .{});
            try writer.print("  shapes: {any},", .{self.shape()});
            try writer.print("  strides: {any},", .{self.stride()});
            try writer.print("  contiguous: {}  ", .{self._is_contiguous});
            try writer.print("}}", .{});
        }
    };
}

pub fn cat(layouts: anytype, comptime dim: usize) Layout(&computeCattedShape(
    @TypeOf(layouts),
    dim,
)) {
    const new_shape = comptime computeCattedShape(@TypeOf(layouts), dim);

    return Layout(&new_shape).init();
}

pub fn stack(layouts: anytype, comptime dim: usize) Layout(&computeStackedShape(@TypeOf(layouts), dim)) {
    const new_shape = comptime computeStackedShape(@TypeOf(layouts), dim);

    return Layout(&new_shape).init();
}

pub fn computePermutedShape(comptime shape_a: []const usize, comptime perm: []const usize) [shape_a.len]usize {
    var new_shape = [_]usize{0} ** shape_a.len;
    for (perm, 0..) |p, idx| {
        for (0..idx) |idx_i| {
            if (perm[idx_i] == perm[idx]) {
                @compileError("perm can't have duplicate idx" ++ std.fmt.comptimePrint("{any}", perm));
            }
        }

        new_shape[idx] = shape_a[p];
    }
    return new_shape;
}

fn computeCattedDims(comptime layouts_type: type) usize {
    switch (@typeInfo(layouts_type)) {
        .@"struct" => |si| {
            if (si.fields.len == 0) @compileError("Empty layouts");

            const s_type = si.fields[0].type;

            if (!@hasDecl(s_type, "IS_LAYOUT")) @compileError("must be a layout");

            const dims = comptime si.fields[0].type.N;

            comptime {
                for (si.fields) |field| {
                    if (field.type.IS_LAYOUT) {
                        if (field.type.N != dims) {
                            @compileError("Layouts must have the same number of dimensions");
                        }
                    }
                }
            }

            return dims;
        },
        else => @compileError("Unsupported type " ++ @typeName(layouts_type)),
    }
}

fn computeCattedShape(comptime layouts_type: type, comptime dim: usize) [computeCattedDims(layouts_type)]usize {
    switch (@typeInfo(layouts_type)) {
        .@"struct" => |si| {
            // const N = comptime computeCattedDims(layouts);
            comptime var base_shape = utils.array.comptimeSliceToArray(si.fields[0].type.S);
            // comptime var base_shape = layouts[0].shape();

            comptime {
                for (si.fields, 0..) |field, f_idx| {
                    if (f_idx == 0) continue;

                    const l_shape = field.type.S;

                    for (l_shape, 0..) |l_dim, idx| {
                        if (idx == dim) {
                            base_shape[idx] += l_dim;
                        } else if (l_dim != base_shape[idx]) {
                            @compileError("Layouts must have the same shape except at dimension " ++ std.fmt.comptimePrint(
                                "{} l_idx: {} l_dim: {} base_shape[idx]: {}",
                                .{ dim, f_idx, l_dim, base_shape[idx] },
                            ));
                        }
                    }
                }
            }

            return base_shape;
        },
        else => @compileError("Unsupported type"),
    }
}

fn computeStackedDims(comptime layouts_type: type) usize {
    switch (@typeInfo(layouts_type)) {
        .@"struct" => |si| {
            if (si.fields.len == 0) @compileError("Empty layouts");

            const s_type = si.fields[0].type;

            if (!@hasDecl(s_type, "IS_LAYOUT")) @compileError("must be a layout");

            comptime {
                for (si.fields) |field| {
                    if (field.type != s_type) @compileError("Layouts must have the same type");
                }
            }

            return s_type.N + 1;
        },
        else => @compileError("Unsupported type"),
    }
}

fn computeStackedShape(comptime layouts_type: type, comptime dim: usize) [computeStackedDims(layouts_type)]usize {
    switch (@typeInfo(layouts_type)) {
        .@"struct" => |si| {
            if (si.fields.len == 0) {
                @compileError("layouts must not be empty");
            }

            const base_shape = utils.array.comptimeSliceToArray(si.fields[0].type.S);
            const N = base_shape.len;

            comptime var shape_i = [_]usize{0} ** (N + 1);

            comptime {
                var i: usize = 0;
                var j: usize = 0;

                while (i < N + 1) {
                    if (i == dim) {
                        shape_i[i] = si.fields.len;
                        i += 1;
                    }

                    shape_i[i] = base_shape[j];
                    i += 1;
                    j += 1;
                }
            }

            return shape_i;
        },
        else => @compileError("Unsupported type"),
    }
}

fn computeTransposedPerm(comptime N: usize, comptime dim0: usize, comptime dim1: usize) [N]usize {
    comptime var perm = [_]usize{0} ** N;
    comptime {
        for (0..N) |i| {
            if (i == dim0) {
                perm[i] = dim1;
            } else if (i == dim1) {
                perm[i] = dim0;
            } else {
                perm[i] = i;
            }
        }
    }

    return perm;
}

fn computeShapeSize(comptime shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }
    return size;
}

fn computeSliceShapeStrides(comptime N: usize, shape: []const usize) [N]usize {
    var new_stride = [_]usize{0} ** N;
    const rank = shape.len;

    // handle zero-dimensional tensor
    if (N == 0) {
        return new_stride;
    }

    var acc: usize = 1;
    var i: usize = rank - 1;
    while (i != 0) : (i -= 1) {
        new_stride[i] = acc;
        acc *= shape[i];
    }
    new_stride[0] = acc;

    return new_stride;
}

test "transpose" {
    const Layout_2x3x4 = Layout(&.{ 2, 3, 4 });
    const layout = Layout_2x3x4.initRaw([_]usize{ 12, 4, 1 });
    const transposed = layout.transpose(0, 2);
    try std.testing.expectEqual([_]usize{ 4, 3, 2 }, transposed.shape());
    try std.testing.expectEqual([_]usize{ 1, 4, 12 }, transposed.stride());
}

test "permute" {
    const Layout3 = Layout(&.{ 2, 3, 4 });
    const layout = Layout3.initRaw([_]usize{ 12, 4, 1 });
    const permuted = layout.permute([_]usize{ 2, 0, 1 });
    try std.testing.expectEqual(
        [_]usize{ 4, 2, 3 },
        permuted.shape(),
    );
    try std.testing.expectEqual([_]usize{ 1, 12, 4 }, permuted.stride());
}

test "reshape" {
    const Layout3 = Layout(&.{ 2, 3, 4 });
    const layout = Layout3.initRaw([_]usize{ 12, 4, 1 });

    const reshaped = layout.reshape(&.{ 6, 4 });
    try std.testing.expectEqual([_]usize{ 6, 4 }, reshaped.shape());
    try std.testing.expectEqual([_]usize{ 4, 1 }, reshaped.stride());
}

test "unsqueeze" {
    const Layout3 = Layout(&.{ 2, 3, 4 });
    const layout = Layout3.initRaw([_]usize{ 12, 4, 1 });

    std.debug.print("layout: {f}\n", .{layout});
    const unsqueezed = layout.unsqueeze(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 3, 4 }, unsqueezed.shape());
    try std.testing.expectEqual([_]usize{ 12, 12, 4, 1 }, unsqueezed.stride());

    const squeezed = unsqueezed.squeeze(1);
    try std.testing.expectEqual([_]usize{ 2, 3, 4 }, squeezed.shape());
    try std.testing.expectEqual([_]usize{ 12, 4, 1 }, squeezed.stride());
    std.debug.print("squeezed: {f}\n", .{squeezed});
}

test "cat or stack" {
    const Layout_2x3x4 = Layout(&.{ 2, 3, 4 });
    const Layout_1x3x4 = Layout(&.{ 1, 3, 4 });

    const l1 = Layout_2x3x4.init();
    const l2 = Layout_1x3x4.init();
    const l3 = Layout_2x3x4.init();

    const lc = cat(.{ l1, l2, l3 }, 0);

    try std.testing.expectEqual([3]usize{ 5, 3, 4 }, lc.shape());

    const ls = stack(.{ l1, l3 }, 2);
    try std.testing.expectEqualDeep([4]usize{ 2, 3, 2, 4 }, ls.shape());
}

pub fn initShapeIterator(arr: anytype) ShapeIterator(utils.array.getArrayShapeComp(@TypeOf(arr))[0]) {
    const NDIM = comptime utils.array.getArrayNDimComp(@TypeOf(arr));
    if (NDIM != 1) @compileError("only support 1-d array");

    const N = comptime utils.array.getArrayShapeComp(@TypeOf(arr))[0];
    const a = @as([N]usize, arr);

    return ShapeIterator(N).init(a);
}

pub fn ShapeIterator(comptime SI: []const usize) type {
    return struct {
        const N = SI.len;

        idx: [N]usize,
        done: bool,

        pub fn init() @This() {
            const idx = [_]usize{0} ** N;

            return @This(){
                .idx = idx,
                .done = false,
            };
        }

        pub fn next(self: *@This()) ?[N]usize {
            if (self.done) return null;

            const outer_indices = self.idx;

            // handle zero-demension
            if (N == 0) {
                self.done = true;
                return self.idx;
            } else {
                var d: usize = N;
                while (d > 0) : (d -= 1) {
                    self.idx[d - 1] += 1;

                    if (self.idx[d - 1] < SI[d - 1]) {
                        break;
                    }
                    self.idx[d - 1] = 0;

                    if (d == 1) self.done = true;
                }

                return outer_indices;
            }
        }
    };
}

test "shape iter" {
    const Layout_2x3x4 = Layout(&.{ 2, 3, 4 });
    // const layout = Layout_2x3x4.init();

    var iter = Layout_2x3x4.iter();

    while (iter.next()) |idx| {
        log.print(@src(), "idx: {any}\n", .{idx});
    }
}
