const std = @import("std");
const utils = @import("./utils.zig");
const log = @import("log.zig");
const shape_expr = @import("shape_expr.zig");

const DimExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;

const product = shape_expr.product;

pub fn Layout(comptime shape_spec: []const DimExpr) type {
    return struct {
        const SE = shape_spec;
        // const S = spec;
        const N = SE.len;
        const IS_LAYOUT = true;

        _shape: [N]usize,
        _stride: [N]usize,
        _shape_env: *const ShapeEnv,
        _is_contiguous: bool = true,

        const Self = @This();

        pub fn broadcastTo(self: Self, comptime target_shape_spec: []const DimExpr) Layout(target_shape_spec) {
            if (comptime shape_expr.shapeExprEqual(SE, target_shape_spec)) {
                return self;
            }

            const new_stride = shape_expr.generateBroadcastStride(SE, self.stride(), target_shape_spec);

            return Layout(target_shape_spec).initRaw(self.shape_env(), new_stride) catch unreachable;
        }

        fn checkContiguous(shape_a: []const usize, strides_a: []const usize) bool {
            var expected_stride: usize = 1;
            var i: usize = N;

            while (i > 0) : (i -= 1) {
                if (N == 0) {
                    return true;
                } else {
                    const dim = shape_a[i - 1];
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

        pub fn init(shape_env_a: *const ShapeEnv) !Self {
            const shape_i = try shape_env_a.lookupShape(SE);
            const strides_i = computeSliceShapeStrides(N, &shape_i);

            return Self.initRaw(shape_env_a, strides_i);
        }

        pub fn initRaw(shape_env_a: *const ShapeEnv, stride_a: [N]usize) !Self {
            const shape_i = try shape_env_a.lookupShape(SE);

            return Self.initInner(shape_env_a, shape_i, stride_a);
        }

        fn initInner(shape_env_a: *const ShapeEnv, shape_a: [N]usize, stride_a: [N]usize) Self {
            const is_contiguous = checkContiguous(&shape_a, &stride_a);

            const layout = Self{
                ._shape = shape_a,
                ._stride = stride_a,
                ._shape_env = shape_env_a,
                ._is_contiguous = is_contiguous,
            };

            return layout;
        }

        pub fn transpose(self: *const Self, comptime dim0: usize, comptime dim1: usize) Layout(&computePermutedShapeExpr(
            SE,
            &computeTransposedPerm(N, dim0, dim1),
        )) {
            if (dim0 >= N or dim1 >= N) @compileError("Invalid dimension");

            const perm = comptime computeTransposedPerm(N, dim0, dim1);

            return self.permute(perm);
        }

        pub fn permute(self: *const Self, comptime perm: [N]usize) Layout(
            &computePermutedShapeExpr(
                SE,
                &perm,
            ),
        ) {
            const new_shape_expr = comptime computePermutedShapeExpr(SE, &perm);
            // std.debug.print("layout permute: old= {any} new= {any}\n", .{ SE, new_shape_expr });

            var new_strides = self._stride;

            inline for (perm, 0..) |p, i| {
                if (p >= N) @compileError("Invalid dimension");

                new_strides[i] = self._stride[p];
            }

            return Layout(&new_shape_expr).initRaw(self._shape_env, new_strides) catch unreachable;
        }

        pub fn reshape(self: *const Self, comptime new_shape_expr: []const DimExpr) !Layout(new_shape_expr) {
            const self_size = comptime if (N == 0) 1 else product(SE);
            const new_size = comptime product(new_shape_expr);

            const ss_v = try self_size.eval(self.shape_env());
            const ns_v = try new_size.eval(self.shape_env());

            if (ss_v != ns_v) {
                return error.InvalidTargetShape;
            }

            return Layout(new_shape_expr).init(self._shape_env) catch unreachable;
        }

        pub fn unsqueeze(self: *const Self, comptime dim: usize) Layout(&shape_expr.insertDimComptime(
            SE,
            dim,
            1,
        )) {
            const new_shape_expr = comptime shape_expr.insertDimComptime(SE, dim, 1);

            return Layout(&new_shape_expr).init(self._shape_env) catch unreachable;
        }

        pub fn squeeze(self: *const Self, comptime dim: usize) Layout(&shape_expr.removeDimComptime(SE, dim)) {
            return Layout(&shape_expr.removeDimComptime(SE, dim)).init(self._shape_env) catch unreachable;
        }

        pub fn iter(self: *const Self) ShapeIterator(N) {
            return ShapeIterator(N).init(self.shape());
        }

        pub fn equal(self: Self, other: Self) bool {
            return std.mem.eql(usize, &self.shape(), &other.shape()) and std.mem.eql(usize, &self.stride(), &other.stride());
        }

        pub fn size(self: *const Self) usize {
            if (N == 0) return 1;
            return utils.product(&self.shape());
        }

        pub fn ndim(_: *const Self) usize {
            return N;
        }

        pub fn shape(self: *const Self) [N]usize {
            return self._shape;
        }

        pub fn shapeRef(self: *const Self) []const usize {
            return &self._shape;
        }

        pub fn shape_env(self: *const Self) *const ShapeEnv {
            return self._shape_env;
        }

        pub fn stride(self: *const Self) [N]usize {
            return self._stride;
        }

        pub fn strideRef(self: *const Self) []const usize {
            return &self._stride;
        }

        pub fn isContiguous(self: *const Self) bool {
            return self._is_contiguous;
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print("Layout {{", .{});
            try writer.print("  shape_expr: {{ ", .{});
            for (SE, 0..) |dim_expr, i| {
                if (i > 0) {
                    try writer.print(", ", .{});
                }
                try writer.print("{f}", .{dim_expr});
            }
            try writer.print(" }}", .{});

            try writer.print("  shapes: {any},", .{self.shape()});
            try writer.print("  strides: {any},", .{self.stride()});
            try writer.print("  contiguous: {}  ", .{self._is_contiguous});
            try writer.print("}}", .{});
        }
    };
}

// pub fn cat(layouts: anytype, comptime dim: usize) Layout(&computeCattedShapeExpr(
//     @TypeOf(layouts),
//     dim,
// )) {
//     const new_shape_expr = comptime computeCattedShapeExpr(@TypeOf(layouts), dim);

//     return Layout(&new_shape_expr).init();
// }

// pub fn stack(layouts: anytype, comptime dim: usize) Layout(&computeStackedShape(@TypeOf(layouts), dim)) {
//     const new_shape = comptime computeStackedShape(@TypeOf(layouts), dim);

//     return Layout(&new_shape).init();
// }

pub fn computePermutedShapeExpr(comptime shape_expr_a: []const DimExpr, comptime perm: []const usize) [shape_expr_a.len]DimExpr {
    var new_shape_expr = [_]DimExpr{undefined} ** shape_expr_a.len;
    for (perm, 0..) |p, idx| {
        for (0..idx) |idx_i| {
            if (perm[idx_i] == perm[idx]) {
                @compileError("perm can't have duplicate idx" ++ std.fmt.comptimePrint("{any}", perm));
            }
        }

        new_shape_expr[idx] = shape_expr_a[p];
    }
    return new_shape_expr;
}

fn isAllStatic(comptime s: []const ?usize) bool {
    for (s) |dim| {
        if (dim == null) return false;
    }
    return true;
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

fn computeCattedShapeExpr(comptime layouts_type: type, comptime dim: usize) [computeCattedDims(layouts_type)]DimExpr {
    switch (@typeInfo(layouts_type)) {
        .@"struct" => |si| {
            // const N = comptime computeCattedDims(layouts);
            _ = dim;
            const base_shape_expr: [computeCattedDims(layouts_type)]DimExpr = utils.array.comptimeSliceToArray(DimExpr, si.fields[0].type.SE);
            // comptime var base_shape = layouts[0].shape();

            // comptime {
            //     for (si.fields, 0..) |field, f_idx| {
            //         if (f_idx == 0) continue;

            //         const l_shape = field.type.SE;

            //         for (l_shape, 0..) |l_dim, idx| {
            //             if (idx == dim) {
            //                 base_shape_expr[idx] = DimExpr.add(&base_shape_expr[idx], &l_dim);
            //             } else if (!l_dim.equal(base_shape_expr[idx])) {
            //                 @compileError("Layouts must have the same shape except at dimension " ++ std.fmt.comptimePrint(
            //                     "{} l_idx: {} l_dim: {} base_shape[idx]: {}",
            //                     .{ dim, f_idx, l_dim, base_shape_expr[idx] },
            //                 ));
            //             }
            //         }
            //     }
            // }

            return base_shape_expr;
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
    const allocator = std.testing.allocator;

    const ddd_dim = comptime shape_expr.SizeExpr.sym(.{ .name = "dddd" });

    const Layout_2x3x4 = Layout(&shape_expr.parseSpec(.{ 2, 3, 4, ddd_dim }));

    var shape_env = ShapeEnv.init(allocator);
    defer shape_env.deinit();
    try shape_env.bind(ddd_dim.Sym.id, 5);

    const layout = try Layout_2x3x4.init(&shape_env);

    std.debug.print("Layout: {f}\n", .{layout});

    const transposed = layout.transpose(0, 2);
    try std.testing.expectEqual([_]usize{ 4, 3, 2, 5 }, transposed.shape());
    try std.testing.expectEqual([_]usize{ 5, 20, 60, 1 }, transposed.stride());
}

test "permute" {
    const allocator = std.testing.allocator;

    const Layout3 = Layout(&shape_expr.parseSpec(.{ 2, 3, "hello" }));

    var shape_env = ShapeEnv.init(allocator);
    defer shape_env.deinit();
    try shape_env.bind(Layout3.SE[2].Sym.id, 4);

    const layout = try Layout3.init(&shape_env);
    const permuted = layout.permute([_]usize{ 2, 0, 1 });
    try std.testing.expectEqual(
        [_]usize{ 4, 2, 3 },
        permuted.shape(),
    );
    try std.testing.expectEqual([_]usize{ 1, 12, 4 }, permuted.stride());
}

test "reshape" {
    const allocator = std.testing.allocator;

    const Layout3 = Layout(&shape_expr.parseSpec(.{ 2, 3, 4 }));
    const layout = try Layout3.init(&ShapeEnv.init(allocator));

    const reshaped = layout.reshape(&shape_expr.parseSpec(.{ 6, 4 }));
    try std.testing.expectEqual([_]usize{ 6, 4 }, reshaped.shape());
    try std.testing.expectEqual([_]usize{ 4, 1 }, reshaped.stride());
}

test "unsqueeze" {
    const allocator = std.testing.allocator;

    const Layout3 = Layout(&shape_expr.parseSpec(.{ 2, 3, 4 }));
    const layout = try Layout3.init(&ShapeEnv.init(allocator));

    std.debug.print("layout: {f}\n", .{layout});
    const unsqueezed = layout.unsqueeze(1);
    try std.testing.expectEqual([_]usize{ 2, 1, 3, 4 }, unsqueezed.shape());
    try std.testing.expectEqual([_]usize{ 12, 12, 4, 1 }, unsqueezed.stride());

    const squeezed = unsqueezed.squeeze(1);
    try std.testing.expectEqual([_]usize{ 2, 3, 4 }, squeezed.shape());
    try std.testing.expectEqual([_]usize{ 12, 4, 1 }, squeezed.stride());
    std.debug.print("squeezed: {f}\n", .{squeezed});
}

// test "cat or stack" {
//     const allocator = std.testing.allocator;

//     const Layout_2x3x4 = Layout(&shape_expr.parseSpec(.{ 2, 3, 4 }));
//     const Layout_1x3x4 = Layout(&shape_expr.parseSpec(.{ 1, 3, 4 }));

//     const shape_env = ShapeEnv.init(allocator);

//     const l1 = try Layout_2x3x4.init(&shape_env);
//     const l2 = try Layout_1x3x4.init(&shape_env);
//     const l3 = try Layout_2x3x4.init(&shape_env);

//     const lc = cat(.{ l1, l2, l3 }, 0);

//     try std.testing.expectEqual([3]usize{ 5, 3, 4 }, lc.shape());

//     const ls = stack(.{ l1, l3 }, 2);
//     try std.testing.expectEqualDeep([4]usize{ 2, 3, 2, 4 }, ls.shape());
// }

pub fn ShapeIterator(comptime N: usize) type {
    return struct {
        shape: [N]usize,
        idx: [N]usize,
        done: bool,

        pub fn init(shape_a: [N]usize) @This() {
            const idx = [_]usize{0} ** N;

            return @This(){
                .shape = shape_a,
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

                    if (self.idx[d - 1] < self.shape[d - 1]) {
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
    const allocator = std.testing.allocator;

    const Layout_2x3x4 = Layout(&shape_expr.parseSpec(.{ 2, 3, 4 }));
    const layout = try Layout_2x3x4.init(&ShapeEnv.init(allocator));

    var iter = layout.iter();

    while (iter.next()) |idx| {
        std.debug.print("idx: {any}\n", .{idx});
    }
}
