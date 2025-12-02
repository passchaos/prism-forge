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

pub fn toOptionalShape(comptime shape: []const usize) [shape.len]?usize {
    var tmp: [shape.len]?usize = undefined;
    inline for (shape, 0..) |val, i| {
        tmp[i] = val; // 自动提升为 ?usize
    }
    return tmp;
}

pub fn printOptional(writer: anytype, comptime fmt: []const u8, value: anytype) !void {
    if (value) |v| {
        try writer.print(fmt, .{v});
    } else {
        try writer.print("null", .{});
    }
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

pub fn getDims(comptime T: type) []const usize {
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

pub fn product(comptime arr: []const usize) usize {
    var result: usize = 1;
    inline for (arr) |dim| {
        result *= dim;
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
