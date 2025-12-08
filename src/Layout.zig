const std = @import("std");
const DataType = @import("./dtype.zig").DataType;
const utils = @import("./utils.zig");
const product = utils.product;

allocator: std.mem.Allocator,
_dtype: DataType,
_shapes: std.ArrayList(usize),
_strides: std.ArrayList(usize),
_transposed: bool = false,

const Self = @This();

pub fn init(allocator: std.mem.Allocator, dt: DataType, shapes_a: std.ArrayList(usize)) !Self {
    const strides_i = try utils.computeStrides(allocator, shapes_a);

    return Self.initRaw(allocator, dt, shapes_a, strides_i);
}

pub fn initRaw(allocator: std.mem.Allocator, dt: DataType, shapes_a: std.ArrayList(usize), strides_a: std.ArrayList(usize)) !Self {
    const layout = Self{
        ._dtype = dt,
        ._shapes = shapes_a,
        ._strides = strides_a,
        .allocator = allocator,
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

    return Self{
        ._dtype = self._dtype,
        ._shapes = new_shapes,
        ._strides = new_strides,
        .allocator = self.allocator,
        ._transposed = true,
    };
}

pub fn reshape(self: *const Self, new_shapes: std.ArrayList(usize)) !Self {
    const new_size = product(new_shapes.items);

    if (new_size != self.size()) {
        return error.InvalidShape;
    }

    var new_strides = try utils.computeStrides(self.allocator, new_shapes);
    if (self._transposed) {
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
