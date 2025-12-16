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

pub const ShapeIterator = struct {
    _shapes: []const usize,

    idx: []usize,
    done: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, shapes_a: []const usize) !@This() {
        var idx = try allocator.alloc(usize, shapes_a.len);
        for (idx, 0..) |_, i| idx[i] = 0;

        return @This(){
            ._shapes = shapes_a,
            .idx = idx,
            .done = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.idx);
    }

    pub fn next(self: *@This()) ?[]const usize {
        if (self.done) return null;

        const outer_indices = self.allocator.alloc(usize, self._shapes.len) catch unreachable;
        @memcpy(outer_indices, self.idx);

        var d: usize = self._shapes.len;

        // handle zero-demension
        if (d == 0) {
            self.done = true;
            return outer_indices;
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

pub fn init(allocator: std.mem.Allocator, dt: DataType, shapes_a: []const usize) !Self {
    const strides_i = try utils.computeStrides(allocator, shapes_a);

    var shapes_i = try std.ArrayList(usize).initCapacity(allocator, shapes_a.len);
    try shapes_i.appendSlice(allocator, shapes_a);

    return Self.initRaw(allocator, dt, shapes_i, strides_i);
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

pub fn transpose(self: *const Self, dim0: usize, dim1: usize) !Self {
    var perm = try self.allocator.alloc(usize, self.ndim());

    for (perm, 0..) |*p, i| p.* = i;

    const tmp = perm[dim0];
    perm[dim0] = perm[dim1];
    perm[dim1] = tmp;

    return try self.permute(perm);
}

pub fn permute(self: *const Self, perm: []const usize) !Self {
    const new_shapes = try self._shapes.clone(self.allocator);
    const new_strides = try self._strides.clone(self.allocator);

    for (perm, 0..) |p, i| {
        new_shapes.items[i] = self._shapes.items[p];
        new_strides.items[i] = self._strides.items[p];
    }

    const is_contiguous = checkContiguous(new_shapes.items, new_strides.items);
    return Self{
        ._dtype = self._dtype,
        ._shapes = new_shapes,
        ._strides = new_strides,
        .allocator = self.allocator,
        ._is_contiguous = is_contiguous,
    };
}

pub fn reshape(self: *const Self, new_shapes: []const usize) !Self {
    const new_size = product(new_shapes);

    if (new_size != self.size()) {
        return error.InvalidShape;
    }

    var shape_list = try std.ArrayList(usize).initCapacity(self.allocator, new_shapes.len);
    try shape_list.appendSlice(self.allocator, new_shapes);

    const new_strides = try utils.computeStrides(self.allocator, new_shapes);

    return Self{
        ._dtype = self._dtype,
        ._shapes = shape_list,
        ._strides = new_strides,
        .allocator = self.allocator,
    };
}

pub fn clone(self: *const Self) !Self {
    return Self{
        ._dtype = self._dtype,
        ._shapes = try self._shapes.clone(self.allocator),
        ._strides = try self._strides.clone(self.allocator),
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
