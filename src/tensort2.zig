const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");
const log = @import("./log.zig");

pub const DataType = enum {
    f32,
    f64,
    i32,
    usize,
    bool,

    const Self = @This();

    pub fn fromAnyType(comptime T: type) Self {
        return switch (T) {
            f32 => .f32,
            f64 => .f64,
            i32 => .i32,
            usize => .usize,
            bool => .bool,
            else => @compileError("Unsupported type"),
        };
    }

    pub fn toType(self: Self) type {
        return switch (self) {
            .f32 => f32,
            .f64 => f64,
            .i32 => i32,
            .usize => usize,
            .bool => bool,
            else => @compileError("Unsupported type"),
        };
    }

    pub fn byteSize(self: Self) usize {
        return switch (self) {
            .f32 => 4,
            .f64 => 8,
            .i32 => 4,
            .usize => 8,
            .bool => 1,
        };
    }
};

fn computeCompStride(comptime shape: []const usize) []const usize {
    const N = shape.len;

    var stride: [N]usize = .{0} ** N;

    var idx = N - 1;
    stride[idx] = 1;

    while (idx > 0) {
        const axis_len = shape[idx];
        stride[idx - 1] = axis_len * stride[idx];

        idx -= 1;
    }

    return stride;
}

fn computeStride(shape: [MAX_RANK]usize, rank: usize) ![MAX_RANK]usize {
    var stride: [MAX_RANK]usize = .{0} ** MAX_RANK;

    var idx = rank - 1;
    stride[idx] = 1;

    while (idx > 0) {
        const axis_len = shape[idx];
        stride[idx - 1] = axis_len * stride[idx];

        idx -= 1;
    }

    return stride;
}

test "computeStride contiguous" {
    const shape = [_]usize{ 2, 3, 4, 0, 0, 0, 0, 0 };
    const rank = 3;
    const stride = try computeStride(shape, rank);

    try std.testing.expectEqualSlices(usize, &.{ 12, 4, 1 }, stride[0..rank]);
    std.debug.print("stride: {any}\n", .{stride});
}

const MAX_RANK: usize = 8;

// Tensor: 动态数据类型、动态shape
// TypeTensor: 静态数据类型、动态shape
// TypeRankTensor: 静态数据类型、静态rank
// ComptimeTensor: 静态数据类型、编译期shape
// StaticTensor: 静态数据类型、静态shape
pub const Tensor = struct {
    buf: []u8,
    shape: [MAX_RANK]usize,
    stride: [MAX_RANK]usize,
    rank: usize,
    offset: usize = 0,
    data_type: DataType,
    owned: bool = false,

    const Self = @This();

    pub fn asTypeTensor(self: Self, comptime T: type) !TypeTensor(T) {
        if (DataType.fromAnyType(T) != self.data_type) return error.TypeMismatch;

        const aligned: [*]align(@sizeOf(T)) u8 = @alignCast(self.buf.ptr);

        const data_ptr: [*]T = @ptrCast(aligned);
        return .{
            .data = data_ptr[0..self.buf.len],
            .shape = self.shape,
            .stride = self.stride,
            .rank = self.rank,
            .offset = self.offset,
            .owned = false,
        };
    }
};

pub fn TypeTensor(comptime T: type) type {
    return struct {
        data: []T,
        shape: [MAX_RANK]usize,
        stride: [MAX_RANK]usize,
        rank: usize,
        offset: usize = 0,
        owned: bool = false,

        const Self = @This();

        pub fn asTensor(self: Self) Tensor {
            return .{
                .buf = std.mem.sliceAsBytes(self.data),
                .shape = self.shape,
                .stride = self.stride,
                .rank = self.rank,
                .offset = self.offset,
                .data_type = DataType.fromAnyType(T),
                .owned = false,
            };
        }

        pub fn asTypeRankTensor(self: Self, comptime N: usize) !TypeRankTensor(T, N) {
            if (N != self.rank) return error.ShapeTooLarge;

            return .{
                .data = self.data,
                .shape = self.shape[0..N],
                .stride = self.stride[0..N],
                .offset = self.offset,
                .owned = false,
            };
        }
    };
}

pub fn fromOwnedData(
    comptime T: type,
    data: []T,
    shape: []usize,
) !TypeTensor(T) {
    if (shape.len > MAX_RANK) return error.ShapeTooLarge;

    var shape_i: [MAX_RANK]usize = .{0} ** MAX_RANK;
    for (0..shape.len) |i| {
        shape_i[i] = shape[i];
    }

    const stride_i = try computeStride(shape_i, shape.len);

    return .{
        .data = data,
        .shape = shape_i,
        .stride = stride_i,
        .rank = shape.len,
        .owned = true,
    };
}

pub fn TypeRankTensor(comptime T: type, comptime N: usize) type {
    return struct {
        data: []T,
        shape: [N]usize,
        stride: [N]usize,
        offset: usize = 0,
        owned: bool = false,

        const Self = @This();

        pub fn asTypeTensor(self: Self) TypeTensor(T) {
            if (N > MAX_RANK) @compileError("Shape too large");
            const shape_i = .{0} ** MAX_RANK;
            const stride_i = .{0} ** MAX_RANK;

            for (0..N) |i| {
                shape_i[i] = self.shape[i];
                stride_i[i] = self.stride[i];
            }

            return .{
                .data = self.data,
                .shape = shape_i,
                .stride = stride_i,
                .rank = N,
                .offset = self.offset,
                .owned = self.owned,
            };
        }

        pub fn asTensor(self: Self) Tensor {
            const t_t = self.asTypeTensor();
            return t_t.asTensor();
        }
    };
}

test "tensor fromData" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var shape = [_]usize{ 2, 3, 4 };

    const type_tensor = try fromOwnedData(f32, data[0..], &shape);
    const tensor = type_tensor.asTensor();

    const tt_tensor = try tensor.asTypeTensor(f32);
    std.debug.print("tt_tensor: {}\n", .{tt_tensor.rank});
}

// 核心设计：
// Tensor: 完全动态张量
//
