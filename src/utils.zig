const std = @import("std");

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
    return comptime isFloat(T) or isInt(T);
}

pub fn isFloat(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float => true,
        else => false,
    };
}

pub fn isInt(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => true,
        else => false,
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

pub fn insertDim(comptime shape: []const ?usize, comptime dim: usize) []const ?usize {
    if (dim > shape.len) @compileError("dim is out of range");

    const new_shape_len = shape.len + 1;
    var tmp: [new_shape_len]?usize = undefined;

    var i = 0;
    while (i < new_shape_len) : (i += 1) {
        if (i == dim) {
            tmp[i] = 1;
            i += 1;
        } else {
            tmp[i] = shape[i];
        }
    }

    return tmp;
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

pub fn toArray(comptime slice: []const usize) [slice.len]usize {
    var result: [slice.len]usize = undefined;
    inline for (slice, 0..) |dim, i| {
        result[i] = dim;
    }
    return result;
}

fn dimsHelper(comptime T: type) []const usize {
    const info = @typeInfo(T);

    if (info == .array) {
        return &[_]usize{info.array.len} ++ dimsHelper(info.array.child);
    } else {
        return &[_]usize{};
    }
}

fn elementType(comptime T: type) type {
    // @compileLog("type: " ++ @typeName(T));
    return switch (@typeInfo(T)) {
        .array => |info| elementType(info.child),
        else => T,
    };
}

pub fn getArrayRefItemType(comptime Ptr: type) type {
    const Deref = @typeInfo(Ptr).pointer.child;
    return elementType(Deref);
}

pub fn flattenAsBytes(comptime T: type, arr: T) []const u8 {
    const info = @typeInfo(T);
    switch (info) {
        .pointer => |ptr| switch (@typeInfo(ptr.child)) {
            .array => |a| {
                const child = a.child;
                if (@typeInfo(child) == .array) {
                    // 多维数组，递归展开
                    const flat: [*]child = @ptrCast(@constCast(arr));
                    @compileLog("flat info: child= {} flat= {}\n", .{ @typeInfo(child), @TypeOf(flat) });
                    return flattenAsBytes(child, flat[0..a.len]);
                } else {
                    // 一维数组，直接转
                    return std.mem.asBytes(arr);
                }
            },
            else => @compileError("Expected array type" ++ std.fmt.comptimePrint("{}", .{info})),
        },
        else => @compileError("Expected pointer type" ++ std.fmt.comptimePrint("{}", .{info})),
    }
}

pub fn getArrayRefShapes(comptime T: type) []const usize {
    switch (@typeInfo(T)) {
        .pointer => |p| switch (p.size) {
            .one => switch (@typeInfo(p.child)) {
                .array => return comptime dimsHelper(p.child),
                .pointer => |pp| switch (pp.size) {
                    .slice => return comptime dimsHelper(pp.child),
                    else => @compileError("Unsupported pointer type"),
                },
                else => @compileError("support only array pointer"),
            },
            else => @compileError("support only array pointer"),
        },
        else => @compileError("support only array pointer"),
    }
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

pub fn computeStrides(allocator: std.mem.Allocator, dims: []const usize) !std.ArrayList(usize) {
    const rank = dims.len;
    var dyn_strides = try std.ArrayList(usize).initCapacity(allocator, rank);

    // handle zero-dimensional tensor
    if (rank == 0) {
        return dyn_strides;
    }

    try dyn_strides.appendNTimes(allocator, 0, rank);

    var acc: usize = 1;
    var i: usize = rank - 1;
    while (i != 0) : (i -= 1) {
        dyn_strides.items[i] = acc;
        acc *= dims[i];
    }
    dyn_strides.items[0] = acc;

    return dyn_strides;
}

pub fn indices_to_flat(indices: []const usize, shape: []const usize, strides_a: []const usize) anyerror!usize {
    if (indices.len == 0) {
        return 0;
        // return error.EmptyIndices;
    }

    var flat_index: usize = 0;
    for (indices, shape, 0..) |index, dim, idx| {
        if (index >= dim) {
            return error.OutOfBounds;
        }

        flat_index += index * strides_a[idx];
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

pub fn cartesianProduct(allocator: std.mem.Allocator, dims: []const usize) !std.ArrayList(std.ArrayList(usize)) {
    const total = product(dims);
    var result = try std.ArrayList(std.ArrayList(usize)).initCapacity(allocator, total);

    const strides = try computeStrides(allocator, dims);

    for (0..total) |i| {
        const indices = try flat_to_indices(allocator, i, strides.items);
        try result.append(allocator, indices);
    }

    return result;
}

pub fn broadcastShapes(allocator: std.mem.Allocator, lhs_shape: []const usize, rhs_shape: []const usize) ![]const usize {
    const l_l = lhs_shape.len;
    const r_l = rhs_shape.len;

    if (l_l == 0 or r_l == 0) {
        return error.InvalidShape;
    }

    const result_len = @max(l_l, r_l);
    var result = try std.ArrayList(usize).initCapacity(result_len);
    try result.appendNTimes(allocator, 0, result_len);

    for (0..result_len) |i| {
        const v = if (l_l > i) blk: {
            const v = if (r_l > i) @max(lhs_shape[l_l - i - 1], rhs_shape[r_l - i - 1]) else lhs_shape[l_l - i - 1];
            break :blk v;
        } else rhs_shape[r_l - i - 1];

        result[result_len - i - 1] = v;
    }

    return result.toOwnedSlice();
}

test "cartesian product" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const res = try cartesianProduct(allocator, &.{ 2, 3, 5 });
    for (res.items) |item| {
        std.debug.print("item: {any}\n", .{item});
    }
}
