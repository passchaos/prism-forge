const std = @import("std");
const log = @import("log.zig");
const storage = @import("storage.zig");

pub const stt = struct {
    pub fn getFieldsLenComptime(comptime T: type) usize {
        return @typeInfo(T).@"struct".fields.len;
    }
};

pub const array = struct {
    pub fn comptimeSliceToArray(comptime S: []const usize) [S.len]usize {
        var result: [S.len]usize = undefined;
        for (S, 0..) |s, i| {
            result[i] = s;
        }
        return result;
    }

    pub fn getArrayNDimComp(comptime T: type) usize {
        switch (@typeInfo(T)) {
            .array => |arr| {
                const child_dims = comptime dimsHelper(arr.child);
                return 1 + child_dims.len;
            },
            else => return 0,
        }
    }

    pub fn getArrayShapeComp(comptime T: type) [getArrayNDimComp(T)]usize {
        switch (@typeInfo(T)) {
            .array => return dimsHelper(T),
            else => @compileError("Unsupported type" ++ @typeName(T)),
        }
    }

    pub fn getArrayShapeCompWithDepth(comptime T: type, comptime D: usize) [D]usize {
        switch (@typeInfo(T)) {
            .array => return dimsHelperWithDepth(T, D),
            else => @compileError("Unsupported type" ++ @typeName(T)),
        }
    }

    pub fn getArrayElementCountComp(comptime T: type) usize {
        const shape = getArrayShapeComp(T);
        return product(&shape);
    }

    pub fn getArrayElementCountCompWithDepth(comptime T: type, comptime D: usize) usize {
        const shape = getArrayShapeCompWithDepth(T, D);
        return product(&shape);
    }

    pub fn getArrayItemTypeComp(comptime T: type) type {
        return switch (@typeInfo(T)) {
            .array => return typeHelper(T),
            else => @compileError("Unsupported type" ++ @typeName(T)),
        };
    }

    pub fn getArrayItemTypeCompWithDepth(comptime T: type, comptime D: usize) type {
        return switch (@typeInfo(T)) {
            .array => return typeHelperWithDepth(T, D),
            else => @compileError("Unsupported type" ++ @typeName(T)),
        };
    }

    fn dimsHelper(comptime T: type) [getArrayNDimComp(T)]usize {
        switch (@typeInfo(T)) {
            .array => |arr| {
                const child_dims = comptime dimsHelper(arr.child);
                return [_]usize{arr.len} ++ child_dims;
            },
            else => return [_]usize{},
        }
    }

    fn dimsHelperWithDepth(comptime T: type, comptime D: usize) [D]usize {
        switch (@typeInfo(T)) {
            .array => |arr| {
                if (D == 1) return [_]usize{arr.len};
                const child_dims = comptime dimsHelperWithDepth(arr.child, D - 1);
                return [_]usize{arr.len} ++ child_dims;
            },
            else => return [_]usize{},
        }
    }

    fn typeHelper(comptime T: type) type {
        return switch (@typeInfo(T)) {
            .array => |arr| return typeHelper(arr.child),
            else => |_| return T,
        };
    }

    fn typeHelperWithDepth(comptime T: type, depth: usize) type {
        return switch (@typeInfo(T)) {
            .array => |arr| {
                if (depth == 1) {
                    return arr.child;
                } else {
                    return typeHelperWithDepth(arr.child, depth - 1);
                }
            },
            else => |_| return T,
        };
    }

    pub fn removeDimComptime(comptime N: usize, comptime arr: []const usize, comptime dim: usize) [N - 1]usize {
        if (dim >= N) @compileError("Invalid dimension");
        if (N == 0) @compileError("don't support 0-1-d tensor removeDim op");
        if (N == 1) return [0]usize{};

        if (N == 0) {
            @compileError("don't support 0-1-d tensor removeDim op");
        } else if (N == 1) {
            return [0]usize{};
        } else {
            if (arr[dim] != 1) @compileError("Dim not one");
            var new_arr = [_]usize{0} ** (N - 1);
            var i: usize = 0;
            var j: usize = 0;

            while (i < N) {
                if (i == dim) {
                    i += 1;
                    continue;
                }
                new_arr[j] = arr[i];
                i += 1;
                j += 1;
            }

            return new_arr;
        }
    }
    pub fn removeDim(comptime N: usize, arr: [N]usize, dim: usize) ![N - 1]usize {
        if (dim >= N) {
            return error.InvalidDim;
        }

        if (N == 0) {
            @compileError("don't support 0-1-d tensor removeDim op");
        } else if (N == 1) {
            return [0]usize{};
        } else {
            var new_arr = [_]usize{0} ** (N - 1);
            var i: usize = 0;
            var j: usize = 0;

            while (i < N) {
                if (i == dim) {
                    i += 1;
                    continue;
                }
                new_arr[j] = arr[i];
                i += 1;
                j += 1;
            }

            return new_arr;
        }
    }

    pub fn insertDimComptime(comptime N: usize, comptime arr: []const usize, comptime dim: usize, comptime value: usize) [N + 1]usize {
        if (dim > N) @compileError("Invalid dimension");

        if (N == 0) {
            return [_]usize{value};
        } else {
            var new_arr = [_]usize{0} ** (N + 1);
            var i: usize = 0;
            var j: usize = 0;

            while (i < N + 1) {
                if (i == dim) {
                    new_arr[i] = value;
                    i += 1;
                    continue;
                }
                new_arr[i] = arr[j];
                i += 1;
                j += 1;
            }

            return new_arr;
        }
    }

    pub fn insertDim(comptime N: usize, arr: [N]usize, dim: usize, value: usize) ![N + 1]usize {
        if (dim > N) {
            return error.InvalidDim;
        }

        if (N == 0) {
            return [_]usize{value};
        } else {
            var new_arr = [_]usize{0} ** (N + 1);
            var i: usize = 0;
            var j: usize = 0;

            while (i < N + 1) {
                if (i == dim) {
                    new_arr[i] = value;
                    i += 1;
                    continue;
                }
                new_arr[i] = arr[j];
                i += 1;
                j += 1;
            }

            return new_arr;
        }
    }
};

pub const tensor = struct {
    pub fn computeTensorsElementType(comptime tensors_type: type) type {
        switch (@typeInfo(tensors_type)) {
            .@"struct" => |si| {
                const base_type = comptime si.fields[0].type.T;

                comptime {
                    for (si.fields) |field| {
                        if (field.type.T != base_type) {
                            @compileError("Layouts must have the same element type");
                        }
                    }
                }

                return base_type;
            },
            else => @compileError("Unsupported type " ++ @typeName(tensors_type)),
        }
    }
    pub fn computeCattedTensorShapeLen(comptime tensors_type: type) usize {
        switch (@typeInfo(tensors_type)) {
            .@"struct" => |si| {
                if (si.fields.len == 0) @compileError("Empty layouts");

                const dims = comptime si.fields[0].type.N;

                comptime {
                    for (si.fields) |field| {
                        if (field.type.N != dims) {
                            @compileError("Layouts must have the same number of dimensions");
                        }
                    }
                }

                return dims;
            },
            else => @compileError("Unsupported type"),
        }
    }
    pub fn computeCattedTensorShape(comptime tensors_type: type, comptime dim: usize) [computeCattedTensorShapeLen(tensors_type)]usize {
        switch (@typeInfo(tensors_type)) {
            .@"struct" => |si| {
                comptime var base_shape = array.comptimeSliceToArray(si.fields[0].type.S);

                comptime {
                    for (si.fields, 0..) |tensor_i, l_idx| {
                        if (l_idx == 0) continue;
                        const l_shape = array.comptimeSliceToArray(tensor_i.type.S);

                        for (l_shape, 0..) |l_dim, idx| {
                            if (idx == dim) {
                                base_shape[idx] += l_dim;
                            } else if (l_dim != base_shape[idx]) {
                                @compileError("Layouts must have the same shape except at dimension " ++ std.fmt.comptimePrint(
                                    "{} l_idx: {} l_dim: {} base_shape[idx]: {}",
                                    .{ dim, l_idx, l_dim, base_shape[idx] },
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

    pub fn computeStackedTensorShapeLen(comptime tensors_type: type) usize {
        switch (@typeInfo(tensors_type)) {
            .@"struct" => |si| {
                if (si.fields.len == 0) @compileError("Empty layouts");

                const dims = comptime si.fields[0].type.N;

                comptime {
                    for (si.fields) |field| {
                        if (field.type.N != dims) {
                            @compileError("Layouts must have the same number of dimensions");
                        }
                    }
                }

                return dims + 1;
            },
            else => @compileError("Unsupported type"),
        }
    }
    pub fn computeStackedTensorShape(comptime tensors_type: type, comptime dim: usize) [computeStackedTensorShapeLen(tensors_type)]usize {
        switch (@typeInfo(tensors_type)) {
            .@"struct" => |si| {
                const base_shape = array.comptimeSliceToArray(si.fields[0].type.S);

                comptime {
                    for (si.fields, 0..) |field, idx| {
                        const f_shape = array.comptimeSliceToArray(field.type.S);
                        if (!std.mem.eql(usize, &f_shape, &base_shape)) {
                            @compileError("Layouts must have the same number of dimensions: " ++ std.fmt.comptimePrint("base shape: {any} meet invalid shape: {any} invalid idx: {}\n", .{ base_shape, f_shape, idx }));
                        }
                    }
                }

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
    pub fn computeTensorsStorages(tensors: anytype) [stt.getFieldsLenComptime(@TypeOf(tensors))]storage.Storage(
        computeTensorsElementType(@TypeOf(tensors)),
        .Cpu,
    ) {
        var storages = [_]storage.Storage(
            computeTensorsElementType(@TypeOf(tensors)),
            .Cpu,
        ){undefined} ** stt.getFieldsLenComptime(@TypeOf(tensors));

        switch (@typeInfo(@TypeOf(tensors))) {
            .@"struct" => |si| {
                inline for (0..si.fields.len) |idx| {
                    // switch (@typeInfo(field.type)) {
                    //     .@"struct" => |sii| {
                    //         inline for (sii.fields) |ff| {
                    //             @compileLog("inner has field: {s} " ++ ff.name);
                    //         }
                    //     },
                    //     else => @compileError("Unsupported type"),
                    // }
                    // @compileLog("field type: " ++ @typeName(field.type));
                    storages[idx] = @field(tensors[idx], "storage");
                }
            },
            else => @compileError("Unsupported type"),
        }

        return storages;
    }

    pub fn matmulResultNDimComp(comptime T1: type, comptime T2: type) usize {
        const DIM1 = T1.DIMS;
        const DIM2 = T2.DIMS;

        if (DIM1 == 1) {
            if (DIM2 == 1) {
                return 1;
            } else if (DIM2 == 2) {
                return 1;
            } else {
                return DIM2 - 1;
            }
        } else if (DIM1 == 2) {
            if (DIM2 == 1) {
                return DIM1;
            } else if (DIM2 == 2) {
                return DIM1;
            } else {
                return @max(DIM1, DIM2);
            }
        } else {
            return @max(DIM1, DIM2);
        }
    }

    pub fn matmulResultElementTypeComp(comptime T1: type, comptime T2: type) type {
        const TE1 = T1.T;
        const TE2 = T2.T;

        if (TE1 == f32) {
            if (TE2 == f32) {
                return f32;
            } else if (TE2 == f64) {
                return f64;
            } else {
                @compileError("Unsupported matmul type" ++ @typeName(TE2));
            }
        } else if (TE1 == f64) {
            if (TE2 == f32) {
                return f64;
            } else if (TE2 == f64) {
                return f64;
            } else {
                @compileError("Unsupported matmul type" ++ @typeName(TE2));
            }
        } else {
            @compileError("Unsupported matmul type" ++ @typeName(TE1));
        }
    }

    pub fn tensorArithmeticTypeCast(comptime A: type, comptime B: type) type {
        if (!isNumber(A) or !isNumber(B)) @compileError("only support number type handle");

        const A1 = comptimeNumberTypeEraseComp(A);
        const B1 = comptimeNumberTypeEraseComp(B);

        const type_size_a = @sizeOf(A1);
        const type_size_b = @sizeOf(B1);

        if (isTypeInt(A1)) {
            if (isTypeInt(B1)) {
                if (type_size_a >= type_size_b) {
                    return A1;
                } else {
                    return B1;
                }
            } else {
                // if one is float, result is float
                return B1;
            }
        } else {
            if (isTypeInt(B1)) {
                return A1;
            } else {
                if (type_size_a >= type_size_b) {
                    return A1;
                } else {
                    return B1;
                }
            }
        }
    }

    pub fn isTensor(comptime T: type) bool {
        if (@hasDecl(T, "Tag") and std.mem.eql(u8, T.Tag, "Tensor")) {
            return true;
        }
        return false;
    }
};

pub fn arithmetricTypePromotion(comptime A: type, comptime B: type) type {
    if (!isNumber(A) or !isNumber(B)) @compileError("only support number type handle");

    const A1 = comptimeNumberTypeEraseComp(A);
    const B1 = comptimeNumberTypeEraseComp(B);

    const type_size_a = @sizeOf(A1);
    const type_size_b = @sizeOf(B1);

    if (isTypeInt(A1)) {
        if (isTypeInt(B1)) {
            if (type_size_a >= type_size_b) {
                return A1;
            } else {
                return B1;
            }
        } else {
            // if one is float, result is float
            return B1;
        }
    } else {
        if (isTypeInt(B1)) {
            return A1;
        } else {
            if (type_size_a >= type_size_b) {
                return A1;
            } else {
                return B1;
            }
        }
    }
}

pub fn promoteNumberType(comptime TT: type, value: anytype) TT {
    const VT = @TypeOf(value);

    if (comptime VT == TT) return value;

    if (comptime isTypeFloat(TT)) {
        if (comptime isTypeFloat(VT)) {
            return @floatCast(value);
        } else {
            return @floatFromInt(value);
        }
    } else {
        if (isTypeInt(VT)) {
            return @intCast(value);
        } else {
            return @intFromFloat(value);
        }
    }
}

pub fn approxEqual(comptime T: type, a: T, b: T, relEps: T, absEps: T) bool {
    return std.math.approxEqAbs(T, a, b, absEps) or
        std.math.approxEqRel(T, a, b, relEps);
}

pub fn sliceApproxEqual(comptime T: type, a: []const T, b: []const T, relEps: T, absEps: T) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (!approxEqual(T, x, y, relEps, absEps)) return false;
    return true;
}

test "approx equal" {
    try std.testing.expect(approxEqual(f32, 2.0001, 2.0000999999, 0.000001, 0.0001));
    try std.testing.expect(sliceApproxEqual(f64, &.{ 10.000001, 2.99999999 }, &.{ 10.0, 3.0 }, 0.0003, 0.00001));
}

pub fn isNumber(comptime T: type) bool {
    return comptime isTypeFloat(T) or isTypeInt(T);
}

pub fn isTypeFloat(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float => true,
        .comptime_float => true,
        else => false,
    };
}

pub fn isTypeInt(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => true,
        .comptime_int => true,
        else => false,
    };
}

pub fn isComptimeFloat(comptime T: type) bool {
    return T == comptime_float;
}

pub fn isComptimeInt(comptime T: type) bool {
    return T == comptime_int;
}

pub fn comptimeNumberTypeEraseComp(comptime T: type) type {
    return comptime switch (@typeInfo(T)) {
        .comptime_float => f64,
        .comptime_int => i64,
        else => T,
    };
}

pub fn floatBasicType(comptime T: type) type {
    return comptime switch (@typeInfo(T)) {
        inline .float => |_| T,
        inline .comptime_float => f64,
        inline else => @compileError("only support f32 and f64"),
    };
}

pub fn sliceEqual(comptime T: type, a: []const T, b: []const T) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        const res = x != y;
        // @compileLog("x: {} y: {}\n", .{ x, y });

        if (res) {
            return false;
        }
    }

    return true;
}

pub fn toOptionalShape(comptime shape: []const usize) [shape.len]?usize {
    var tmp: [shape.len]?usize = undefined;
    inline for (shape, 0..) |val, i| {
        tmp[i] = val; // 自动提升为 ?usize
    }
    return tmp;
}

pub fn printOptional(writer: anytype, comptime fmt: []const u8, value: anytype) !void {
    try writer.print(fmt, .{value});
}

pub fn getSliceItemType(comptime Slice: type) type {
    const Deref = @typeInfo(Slice).pointer.child;
    return elementType(Deref);
}

pub fn elementType(comptime T: type) type {
    // @compileLog("type: " ++ @typeName(T));
    return switch (@typeInfo(T)) {
        .array => |info| elementType(info.child),
        else => T,
    };
}

fn ElemOf(comptime V: type) type {
    return switch (@typeInfo(V)) {
        .pointer => |p| switch (p.size) {
            .one => switch (@typeInfo(p.child)) {
                .array => |arr| arr.child,
                .pointer => |pp| switch (pp.size) {
                    .slice => pp.child,
                    else => @compileError("Unsupported pointer type" ++ std.fmt.comptimePrint(" get type: {}", .{@typeInfo(@TypeOf(pp))})),
                },
                else => |v| @compileError(std.fmt.comptimePrint("Unsupported pointer type: info= {} info_child= {}\n", .{ p, v })),
            },
            else => @compileError("Unsupported pointer type"),
        },
        .array => @compileError("ElementOf: use array will get a copy of argument, so can't get valid value"),
        else => @compileError("Unsupported type"),
    };
}

pub fn asSlice(value: anytype) []const ElemOf(@TypeOf(value)) {
    return switch (@typeInfo(@TypeOf(value))) {
        .pointer => |p| switch (p.size) {
            .one => switch (@typeInfo(p.child)) {
                .array => &value.*,
                .pointer => |pp| switch (pp.size) {
                    .slice => value.*,
                    else => @compileError("Unsupported pointer type"),
                },
                else => |v| @compileError(std.fmt.comptimePrint("unsupported pointer to non-array: {}", .{v})),
            },
            else => @compileError("Unsupported pointer type"),
        },
        else => @compileError("Unsupported type, must be pointer"),
    };
}

pub fn product(arr: []const usize) usize {
    var result: usize = 1;
    for (arr) |item| {
        result *= item;
    }
    return result;
}

pub fn allStatic(comptime dims: ?[]const ?usize) bool {
    if (dims == null) return false;

    inline for (dims.?) |dim| {
        if (dim == null) return false;
    }
    return true;
}

pub fn computeArrayShapeStrides(comptime N: usize, shapes: [N]usize) [N]usize {
    var strides = [_]usize{0} ** N;

    if (N > 0) {
        var acc: usize = 1;
        var i: usize = N - 1;
        while (i != 0) : (i -= 1) {
            strides[i] = acc;
            acc *= shapes[i];
        }

        strides[0] = acc;
    }

    return strides;
}

pub fn computeStrides(comptime shape: []const usize) [shape.len]usize {
    const rank = shape.len;
    var new_stride = [_]usize{0} ** rank;

    // handle zero-dimensional tensor
    if (rank == 0) {
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

pub fn indexShapeToFlat(comptime shape_a: []const usize, index: [shape_a.len]usize) !usize {
    const stride = computeStrides(shape_a);

    return try indexToFlat(&index, shape_a, &stride);
}

pub fn indexToFlat(indices: []const usize, shape: []const usize, stride_a: []const usize) anyerror!usize {
    var flat_index: usize = 0;
    for (indices, shape, 0..) |index, dim, idx| {
        if (index >= dim) {
            return error.OutOfBounds;
        }

        flat_index += index * stride_a[idx];
    }
    return flat_index;
}

pub fn flat_to_indices(allocator: std.mem.Allocator, flat_index: usize, strides_a: []const usize) !std.ArrayList(usize) {
    var indices = try std.ArrayList(usize).initCapacity(allocator, strides_a.len);

    var tmp = flat_index;
    for (0..strides_a.len) |dim| {
        try indices.append(allocator, tmp / strides_a[dim]);
        tmp %= strides_a[dim];
    }
    return indices;
}

pub fn generateBroadcastStride(
    comptime N: usize,
    comptime BN: usize,
    orig_shape: [N]usize,
    orig_stride: [N]usize,
    target_shape: [BN]usize,
) ![BN]usize {
    if (N > BN) @compileError("can't broadcast to smaller dimension");

    var new_stride = [_]usize{0} ** BN;

    var old_i: isize = @intCast(orig_shape.len);
    old_i -= 1;
    var new_i: isize = @intCast(target_shape.len);
    new_i -= 1;

    while (new_i >= 0) {
        const n_dim = target_shape[@intCast(new_i)];

        if (old_i >= 0) {
            const old_i_u: usize = @intCast(old_i);

            const o_dim = orig_shape[old_i_u];
            const o_stride = orig_stride[old_i_u];

            if (o_dim == n_dim) {
                new_stride[@intCast(new_i)] = o_stride;
            } else if (o_dim == 1 and n_dim > 1) {
                new_stride[@intCast(new_i)] = 0;
            } else {
                log.print(
                    @src(),
                    "orig_shape: {any} orig_stride: {any} target_shape: {any}\n",
                    .{ orig_shape, orig_stride, target_shape },
                );
                return error.ShapeMismatch;
            }

            old_i -= 1;
            new_i -= 1;
        } else {
            new_stride[@intCast(new_i)] = 0;

            new_i -= 1;
        }
    }

    return new_stride;
}

pub fn compatibleBroacastShapes(comptime N: usize, lhs_shape: [N]usize, rhs_shape: [N]usize) ![N]usize {
    const l_l = lhs_shape.len;
    const r_l = rhs_shape.len;

    if (l_l == 0 or r_l == 0) {
        return error.InvalidShape;
    }

    const result_len = @max(l_l, r_l);

    var result = [_]usize{0} ** result_len;

    for (0..result_len) |i| {
        const v = if (l_l > i) blk: {
            const v = if (r_l > i) @max(lhs_shape[l_l - i - 1], rhs_shape[r_l - i - 1]) else lhs_shape[l_l - i - 1];
            break :blk v;
        } else rhs_shape[r_l - i - 1];

        result[result_len - i - 1] = v;
    }

    return result;
}

test "get array len" {
    const arr = [2][3][4]f32{
        [3][4]f32{
            [4]f32{ 1.0, 2.0, 3.0, 4.0 },
            [4]f32{ 5.0, 6.0, 7.0, 8.0 },
            [4]f32{ 9.0, 10.0, 11.0, 12.0 },
        },
        [3][4]f32{
            [4]f32{ 13.0, 14.0, 15.0, 16.0 },
            [4]f32{ 17.0, 18.0, 19.0, 20.0 },
            [4]f32{ 21.0, 22.0, 23.0, 24.0 },
        },
    };

    const ndim = comptime array.getArrayNDimComp(@TypeOf(arr));
    try std.testing.expectEqual(ndim, 3);

    const element_count = comptime array.getArrayElementCountComp(@TypeOf(arr));
    try std.testing.expectEqual(element_count, 24);

    const shape = comptime array.getArrayShapeComp(@TypeOf(arr));
    const dtype = comptime array.getArrayItemTypeComp(@TypeOf(arr));

    try std.testing.expectEqualSlices(usize, &[3]usize{ 2, 3, 4 }, &shape);
    try std.testing.expect(if (dtype == f32) true else false);

    log.print(@src(), "shape: {any} dtype: {}\n", .{ shape, dtype });

    const s: []const f32 = @ptrCast(&arr);
    try std.testing.expectEqual(s.len, 24);
    log.print(@src(), "s: {any} len: {}\n", .{ s, s.len });
}

test "broadcast shape" {
    const orig_shape = [_]usize{ 1, 3 };
    const target_shape = [_]usize{ 5, 3 };
    const broadcasted_stride = try generateBroadcastStride(
        2,
        2,
        orig_shape,
        computeStrides(2, orig_shape),
        target_shape,
    );
    try std.testing.expectEqual([2]usize{ 0, 1 }, broadcasted_stride);

    log.print(@src(), "begin compatible handle\n", .{});
    const compatible_broadcasted_shape = try compatibleBroacastShapes(
        2,
        orig_shape,
        target_shape,
    );
    try std.testing.expectEqualSlices(usize, &compatible_broadcasted_shape, &target_shape);
}
