const std = @import("std");
const DataType = @import("./dtype.zig").DataType;
const utils = @import("./utils.zig");
const product = utils.product;

allocator: std.mem.Allocator,
_dtype: DataType,
_shapes: std.ArrayList(usize),
_strides: std.ArrayList(usize),
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

pub fn init(allocator: std.mem.Allocator, dt: DataType, shapes_a: std.ArrayList(usize)) !Self {
    const strides_i = try utils.computeStrides(allocator, shapes_a);

    return Self.initRaw(allocator, dt, shapes_a, strides_i);
}

pub fn initRaw(allocator: std.mem.Allocator, dt: DataType, shapes_a: std.ArrayList(usize), strides_a: std.ArrayList(usize)) !Self {
    const is_contiguous = checkContiguous(shapes_a.items, strides_a.items);

    const layout = Self{
        ._dtype = dt,
        ._shapes = shapes_a,
        ._strides = strides_a,
        .allocator = allocator,
        ._is_contiguous = is_contiguous,
    };

    return layout;
}

pub fn transpose(self: *const Self) !Self {
    if (self._shapes.items.len != 2) {
        return error.InvalidShape;
    }

    var new_shapes = try self._shapes.clone(self.allocator);
    var new_strides = try self._strides.clone(self.allocator);

    new_shapes.items[0] = self._shapes.items[1];
    new_shapes.items[1] = self._shapes.items[0];
    new_strides.items[0] = self._strides.items[1];
    new_strides.items[1] = self._strides.items[0];

    const is_contiguous = checkContiguous(new_shapes.items, new_strides.items);
    return Self{
        ._dtype = self._dtype,
        ._shapes = new_shapes,
        ._strides = new_strides,
        .allocator = self.allocator,
        ._is_contiguous = is_contiguous,
    };
}

pub fn reshape(self: *const Self, new_shapes: std.ArrayList(usize)) !Self {
    const new_size = product(new_shapes.items);

    if (new_size != self.size()) {
        return error.InvalidShape;
    }

    var new_strides = try utils.computeStrides(self.allocator, new_shapes);
    if (self._is_contiguous) {
        const tmp0 = new_strides.items[0];
        const tmp1 = new_strides.items[1];
        new_strides.items[0] = tmp1;
        new_strides.items[1] = tmp0;
    }

    return Self{
        ._dtype = self._dtype,
        ._shapes = new_shapes,
        ._strides = new_strides,
        .allocator = self.allocator,
    };
}

pub fn equal(self: *const Self, other: *const Self) bool {
    return self._dtype == other._dtype and std.mem.eql(usize, self._shapes.items, other._shapes.items) and std.mem.eql(usize, self._strides.items, other._strides.items);
}

pub fn size(self: *const Self) usize {
    return product(self._shapes.items);
}

pub fn dtypeSize(self: *const Self) usize {
    return self._dtype.dtypeSize();
}

pub fn dtype(self: *const Self) DataType {
    return self._dtype;
}

pub fn ndim(self: *const Self) usize {
    return self._shapes.items.len;
}

pub fn shapes(self: *const Self) []usize {
    return self._shapes.items;
}

pub fn strides(self: *const Self) []usize {
    return self._strides.items;
}

pub fn isContiguous(self: *const Self) bool {
    return self._is_contiguous;
}

pub fn format(
    self: @This(),
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.print("Layout {{\n", .{});
    try writer.print("  dtype: {},\n", .{self._dtype});
    try writer.print("  shapes: {any},\n", .{self._shapes.items});
    try writer.print("  strides: {any},\n", .{self._strides.items});
    try writer.print("  contiguous: {}\n", .{self._is_contiguous});
    try writer.print("}}\n", .{});
}
