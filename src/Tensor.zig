const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");

const dtype_o = @import("./dtype.zig");
const DataType = dtype_o.DataType;
const Scalar = dtype_o.Scalar;

const Layout = @import("./Layout.zig");

const Storage = @import("./Storage.zig");

const asSlice = utils.asSlice;

const TensorIterator = struct {
    tensor: *const Self,

    idx: []usize,
    done: bool,

    fn init(tensor: *const Self) !@This() {
        var idx = try tensor.allocator.alloc(usize, tensor.ndim());
        for (idx, 0..) |_, i| idx[i] = 0;

        return @This(){
            .tensor = tensor,
            .idx = idx,
            .done = false,
        };
    }

    fn deinit(self: *@This()) void {
        self.tensor.allocator.free(self.idx);
    }

    fn next(self: *@This()) ?usize {
        if (self.done) return null;

        var offset: usize = self.tensor._storage_offset;
        for (self.idx, 0..) |i, d| {
            offset += i * self.tensor.strides()[d];
        }

        var d: usize = self.tensor.ndim();
        while (d > 0) : (d -= 1) {
            self.idx[d - 1] += 1;

            if (self.idx[d - 1] < self.tensor.shapes()[d - 1]) {
                break;
            }
            self.idx[d - 1] = 0;

            if (d == 1) self.done = true;
        }

        return offset;
    }
};

const Self = @This();

allocator: std.mem.Allocator,
storage: Storage,
layout: Layout,
_storage_offset: usize = 0,

pub fn scalarItem(self: *const Self) !Scalar {
    if (self.ndim() == 0) {
        return self.getWithIndices(&.{});
    } else {
        return error.ShapeMismatch;
    }
}

// scope method
pub fn cat(allocator: std.mem.Allocator, tensors: []const Self, dim: usize) !Self {
    const dtype_i = tensors[0].dtype();
    const rank = tensors[0].ndim();

    for (tensors) |t| {
        if (t.dtype() != dtype_i or t.ndim() != rank) return error.ShapeMismatch;
    }

    var new_shape = try allocator.alloc(usize, rank);
    for (0..rank) |i| {
        if (i == dim) {
            var sum_i: usize = 0;
            for (tensors) |t| {
                sum_i += t.shapes()[i];
            }
            new_shape[i] = sum_i;
        } else {
            new_shape[i] = tensors[0].shapes()[i];
        }
    }

    var total: usize = 1;
    for (new_shape) |s| total *= s;

    const elem_size = dtype_i.dtypeSize();
    var new_buf = try allocator.alloc(u8, total * elem_size);

    var offset: usize = 0;
    for (tensors) |t| {
        const bytes = t.rawDataSlice();
        @memcpy(new_buf[offset .. offset + bytes.len], bytes);
        offset += bytes.len;
    }

    return Self.fromDataRaw(allocator, dtype_i, new_shape, Storage.Device.Cpu, @ptrCast(new_buf), new_buf.len);
}

pub fn stack(allocator: std.mem.Allocator, tensors: []const Self, dim: usize) !Self {
    const dtype_i = tensors[0].dtype();
    const rank = tensors[0].ndim();

    for (tensors) |t| {
        if (t.dtype() != dtype_i or !std.mem.eql(usize, t.shapes(), tensors[0].shapes())) return error.ShapeMismatch;
    }

    var new_shape = try allocator.alloc(usize, rank + 1);
    for (0..dim) |i| new_shape[i] = tensors[0].shapes()[i];
    new_shape[dim] = tensors.len;
    for (dim..rank) |i| new_shape[i + 1] = tensors[0].shapes()[i];

    var total: usize = 1;
    for (new_shape) |s| total *= s;

    const elem_size = dtype_i.dtypeSize();
    var new_buf = try allocator.alloc(u8, total * elem_size);

    var offset: usize = 0;
    for (tensors) |t| {
        const bytes = t.rawDataSlice();
        std.debug.print("len: {}\n", .{bytes.len});
        @memcpy(new_buf[offset .. offset + bytes.len], bytes);
        offset += bytes.len;
    }

    return Self.fromDataRaw(allocator, dtype_i, new_shape, Storage.Device.Cpu, @ptrCast(new_buf), new_buf.len);
}

// divide
pub fn split(self: *const Self, chunk_size: usize, dim: usize) ![]const Self {
    if (dim >= self.ndim()) return error.InvalidDim;

    const dim_len = self.shapes()[dim];
    if (chunk_size == 0 or chunk_size > dim_len) return error.InvalidSplit;

    const num_splits = (dim_len + chunk_size - 1) / chunk_size;
    var result = try self.allocator.alloc(Self, num_splits);

    var offset: usize = 0;
    for (0..num_splits) |i| {
        const chunk_size_i = if ((offset + chunk_size) <= dim_len) chunk_size else (dim_len - offset);

        var new_shape = try std.ArrayList(usize).initCapacity(self.allocator, self.shapes().len);
        try new_shape.appendSlice(self.allocator, self.shapes());
        new_shape.items[dim] = chunk_size_i;

        // must use old strides
        var new_strides = try std.ArrayList(usize).initCapacity(self.allocator, self.strides().len);
        try new_strides.appendSlice(self.allocator, self.strides());

        const layout = try Layout.initRaw(self.allocator, self.dtype(), new_shape, new_strides);
        result[i] = try Self.fromDataImpl(self.allocator, layout, self.storage.clone(), self._storage_offset + offset * self.strides()[dim]);

        offset += chunk_size_i;
    }

    return result;
}

pub fn chunk(self: *const Self, chunk_count: usize, dim: usize) ![]const Self {
    if (dim >= self.ndim()) return error.InvalidDim;

    const dim_len = self.shapes()[dim];
    if (chunk_count == 0 or chunk_count > dim_len) return error.InvalidSplit;

    const chunk_size_i = (dim_len + chunk_count - 1) / chunk_count;
    return try self.split(chunk_size_i, dim);
}

pub fn unbind(self: *const Self, dim: usize) ![]const Self {
    if (dim >= self.ndim()) return error.InvalidDim;

    const dim_len = self.shapes()[dim];
    var result = try self.allocator.alloc(Self, dim_len);

    for (0..dim_len) |i| {
        var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, self.ndim() - 1);
        var new_strides = try std.ArrayList(usize).initCapacity(self.allocator, self.ndim() - 1);

        for (0..self.ndim()) |j| {
            if (j == dim) continue;
            try new_shapes.append(self.allocator, self.shapes()[j]);
            try new_strides.append(self.allocator, self.strides()[j]);
        }

        const layout = try Layout.initRaw(self.allocator, self.dtype(), new_shapes, new_strides);
        result[i] = try Self.fromDataImpl(self.allocator, layout, self.storage.clone(), self._storage_offset + i * self.strides()[dim]);
    }
    return result;
}

// elementwise method
pub fn map_(self: *Self, comptime data_type: DataType, func: fn (data_type.toTypeComp()) data_type.toTypeComp()) !void {
    var iter = try self.dataIter();

    const data_slice = self.dataSlice(data_type.toTypeComp());
    while (iter.next()) |idx| {
        const x = &data_slice[idx];
        x.* = func(x.*);
    }
}

pub fn map(self: *const Self, comptime data_type: DataType, func: fn (data_type.toTypeComp()) data_type.toTypeComp()) !Self {
    var a = try self.contiguous();

    try a.map_(data_type, func);
    return a;
}

pub fn binaryOp_(self: *Self, b: Self, comptime data_type: DataType, op_func: fn (x: data_type.toTypeComp(), y: data_type.toTypeComp()) data_type.toTypeComp()) !void {
    // inplace method: need broadcast to self shape
    var b_i = b;
    try b_i.broadcast_to_(self.shapes());

    var iter = try self.dataIter();
    defer iter.deinit();

    while (iter.next()) |idx| {
        const x = &self.dataSlice(data_type.toTypeComp())[idx];
        const y = b_i.dataSlice(data_type.toTypeComp())[idx];
        x.* = op_func(x.*, y);
    }
}

pub fn binaryOp(self: *const Self, b: Self, comptime data_type: DataType, op_func: fn (x: data_type.toTypeComp(), y: data_type.toTypeComp()) data_type.toTypeComp()) !Self {
    const target_shapes = try utils.compatibleBroacastShapes(self.allocator, self.shapes(), b.shapes());

    const a = try self.broadcast_to(target_shapes.items);
    var a1 = try a.contiguous();
    const c = try b.broadcast_to(target_shapes.items);

    try a1.binaryOp_(c, data_type, op_func);

    return a1;
}

pub fn clamp_(self: *Self, min_a: anytype, max_a: anytype) !void {
    const DT = comptime DataType.typeToDataType(@TypeOf(min_a));
    const scope = struct {
        fn call(v: DT.toTypeComp()) DT.toTypeComp() {
            return std.math.clamp(v, min_a, max_a);
        }
    }.call;

    try self.map_(DT, scope);
}

pub fn add_(self: *Self, value: anytype) !void {
    if (@TypeOf(value) == @This()) {
        switch (self.dtype()) {
            inline else => |dt| {
                const scope = struct {
                    fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
                        return v + other;
                    }
                }.call;

                return try self.binaryOp_(@as(@This(), value), dt, scope);
            },
        }
    }

    const DT = comptime DataType.typeToDataType(@TypeOf(value));
    const scope = struct {
        fn call(v: DT.toTypeComp()) DT.toTypeComp() {
            return v + value;
        }
    }.call;

    try self.map_(DT, scope);
}

pub fn add(self: *Self, value: anytype) !Self {
    if (@TypeOf(value) == @This()) {
        switch (self.dtype()) {
            inline else => |dt| {
                const scope = struct {
                    fn call(v: dt.toTypeComp(), other: dt.toTypeComp()) dt.toTypeComp() {
                        return v + other;
                    }
                }.call;

                return try self.binaryOp(@as(@This(), value), dt, scope);
            },
        }
    }

    const DT = comptime DataType.typeToDataType(@TypeOf(value));
    const scope = struct {
        fn call(v: DT.toTypeComp()) DT.toTypeComp() {
            return v + value;
        }
    }.call;

    return try self.map(DT, scope);
}

pub fn sub_(self: *Self, value: anytype) !void {
    const DT = comptime DataType.typeToDataType(@TypeOf(value));
    const scope = struct {
        fn call(v: *DT.toTypeComp()) void {
            v.* -= value;
        }
    }.call;

    try self.map_(DT, scope);
}

pub fn mul_(self: *Self, value: anytype) !void {
    const DT = comptime DataType.typeToDataType(@TypeOf(value));

    const T = DT.toTypeComp();
    const func = struct {
        fn call(v: T) T {
            return v * value;
        }
    }.call;

    try self.map_(DT, func);
}

pub fn div_(self: *Self, value: anytype) !void {
    const DT = comptime DataType.typeToDataType(@TypeOf(value));
    const T = DT.toTypeComp();

    const func = struct {
        fn call(v: T) T {
            return v / value;
        }
    }.call;

    try self.map_(DT, func);
}

pub fn sin_(self: *Self) !void {
    switch (self.dtype()) {
        inline .f16, .f32 => |DT| {
            const func = struct {
                fn call(v: DT.toTypeComp()) DT.toTypeComp() {
                    return @sin(v);
                }
            }.call;
            try self.map_(DT, func);
        },
        inline else => return error.UnsupportedType,
    }
}

pub fn exp_(self: *Self) !void {
    switch (self.dtype()) {
        inline .f16, .f32 => |DT| {
            const func = struct {
                fn call(v: DT.toTypeComp()) DT.toTypeComp() {
                    return @exp(v);
                }
            }.call;
            try self.map_(DT, func);
        },
        inline else => return error.UnsupportedType,
    }
}

//
//
//
// reduce method
pub fn reduce(self: *const Self, comptime data_type: DataType, dim: ?usize, op_init: data_type.toTypeComp(), op_func: fn (acc: data_type.toTypeComp(), x: data_type.toTypeComp()) data_type.toTypeComp(), post_func: ?fn (acc: data_type.toTypeComp(), count: usize) data_type.toTypeComp()) !Self {
    const T = data_type.toTypeComp();

    if (dim) |dm| {
        var shapes_i = try std.ArrayList(usize).initCapacity(self.allocator, self.ndim() - 1);
        try shapes_i.appendSlice(self.allocator, self.shapes());
        _ = shapes_i.orderedRemove(dm);

        const data_len = utils.product(shapes_i.items);
        var new_buf = try self.allocator.alloc(T, data_len);

        var indices = try self.allocator.alloc(usize, self.ndim());
        defer self.allocator.free(indices);

        for (indices) |*i| i.* = 0;

        var out_i: usize = 0;
        var done = false;

        while (!done) {
            var acc: T = op_init;
            for (0..self.shapes()[dm]) |k| {
                indices[dm] = k;
                acc = op_func(acc, (try self.getWithIndicesCompType(data_type, indices)).*);
            }

            if (post_func) |pf| {
                acc = pf(acc, self.shapes()[dm]);
            }

            new_buf[out_i] = acc;
            out_i += 1;

            var j = self.shapes().len - 1;

            if (j == 0) {
                break;
            }

            while (j > 0) : (j -= 1) {
                if (j - 1 == dm) continue;

                indices[j - 1] += 1;
                if (indices[j - 1] < self.shapes()[j - 1]) {
                    break;
                } else if (j == 1) {
                    done = true;
                } else {
                    indices[j - 1] = 0;
                }
            }
        }

        const layout = try Layout.init(self.allocator, data_type, shapes_i.items);

        const bytes_size = new_buf.len * @sizeOf(T);
        const storage = Storage.init(self.allocator, Storage.Device.Cpu, @ptrCast(new_buf.ptr), bytes_size);

        return try Self.fromDataImpl(self.allocator, layout, storage, 0);
    } else {
        var total: T = op_init;

        var idx = try self.allocator.alloc(usize, self.shapes().len);
        for (idx) |*x| x.* = 0;

        var done = false;

        var count: usize = 0;
        while (!done) {
            total = op_func(total, (try self.getWithIndicesCompType(data_type, idx)).*);

            count += 1;

            var d: usize = self.shapes().len;
            while (d > 0) : (d -= 1) {
                idx[d - 1] += 1;

                if (idx[d - 1] < self.shapes()[d - 1]) {
                    break;
                } else if (d == 1) {
                    done = true;
                } else {
                    idx[d - 1] = 0;
                }
            }
        }

        if (post_func) |pf| {
            total = pf(total, count);
        }

        var new_buf = try self.allocator.alloc(T, 1);
        new_buf[0] = total;

        const layout = try Layout.init(self.allocator, data_type, &.{1});
        const storage = Storage.init(self.allocator, Storage.Device.Cpu, @ptrCast(new_buf.ptr), @sizeOf(T));

        return try Self.fromDataImpl(self.allocator, layout, storage, 0);
    }
}

pub fn sum(self: *const Self, comptime data_type: DataType, dim: ?usize) !Self {
    const T = data_type.toTypeComp();
    const scope = struct {
        fn op_func(acc: T, val: T) T {
            return acc + val;
        }
    };
    return try self.reduce(data_type, dim, 0, scope.op_func, null);
}

pub fn max(self: *const Self, comptime data_type: DataType, dim: ?usize) !Self {
    const T = data_type.toTypeComp();
    const scope = struct {
        fn op_func(acc: T, val: T) T {
            return @max(acc, val);
        }
    };
    return try self.reduce(data_type, dim, -std.math.inf(T), scope.op_func, null);
}

pub fn min(self: *const Self, comptime data_type: DataType, dim: ?usize) !Self {
    const T = data_type.toTypeComp();
    const scope = struct {
        fn op_func(acc: T, val: T) T {
            return @min(acc, val);
        }
    };
    return try self.reduce(data_type, dim, std.math.inf(T), scope.op_func, null);
}

pub fn prod(self: *const Self, comptime data_type: DataType, dim: ?usize) !Self {
    const T = data_type.toTypeComp();
    const scope = struct {
        fn op_func(acc: T, val: T) T {
            return acc * val;
        }
    };
    return try self.reduce(data_type, dim, std.math.inf(T), scope.op_func, null);
}

pub fn mean(self: *const Self, comptime data_type: DataType, dim: ?usize) !Self {
    const T = data_type.toTypeComp();

    // std.debug.print("type info: {any}\n", .{@typeInfo())})
    if (@typeInfo(T) != .float) {
        @compileError("only support float");
    }

    const scope = struct {
        fn op_func(acc: T, val: T) T {
            return acc + val;
        }
        fn post_func(acc: T, count: usize) T {
            return acc / @as(T, @floatFromInt(count));
        }
    };

    return try self.reduce(data_type, dim, 0, scope.op_func, scope.post_func);
}

// op method
pub fn matmul(self: *const Self, other: *const Self) anyerror!Self {
    if (self.dtype() != other.dtype()) {
        return error.TypeMismatch;
    }

    if (self.ndim() != 2 or other.ndim() != 2) {
        return error.ShapeMismatch;
    }

    if (self.dtype() != DataType.f32) {
        return error.TypeNotSupported;
    }

    if (self.shapes()[1] != other.shapes()[0]) {
        return error.ShapeMismatch;
    }

    const lhs = if (!self.layout.isContiguous()) &(try self.contiguous()) else self;

    const rhs = if (!other.layout.isContiguous()) &(try other.contiguous()) else other;

    const m = lhs.shapes()[0];
    const n = rhs.shapes()[1];
    const k = lhs.shapes()[1];

    const a: [*c]const f32 = @ptrCast(lhs.storage.dataSlice(f32));
    const b: [*c]const f32 = @ptrCast(rhs.storage.dataSlice(f32));

    const buf = try std.ArrayList(f32).initCapacity(lhs.allocator, m * n);
    const c = @as([*]f32, @ptrCast(buf.items.ptr));

    host.matmul(a, b, c, m, n, k);

    const data = @as([*]u8, @ptrCast(c));

    return try Self.fromDataRaw(lhs.allocator, DataType.f32, &.{ m, n }, Storage.Device.Cpu, data, m * n * @sizeOf(f32));
}

// iterate method
pub fn dataIter(self: *const Self) !TensorIterator {
    return try TensorIterator.init(self);
}

// create method
pub fn fromDataImpl(allocator: std.mem.Allocator, layout_a: Layout, storage_a: Storage, storage_offset_a: usize) !Self {
    const layout_bytes_zie = layout_a.size() * layout_a.dtypeSize();
    if (layout_bytes_zie + storage_offset_a > storage_a.byteSize()) {
        return error.InvalidSize;
    }

    return Self{
        .allocator = allocator,
        .layout = layout_a,
        .storage = storage_a,
        ._storage_offset = storage_offset_a,
    };
}

pub fn fromDataRaw(allocator: std.mem.Allocator, dtype_i: DataType, shapes_a: []const usize, device: Storage.Device, data: [*]u8, bytes_size: usize) anyerror!Self {
    const storage = Storage.init(allocator, device, data, bytes_size);

    const layout = try Layout.init(allocator, dtype_i, shapes_a);

    return try Self.fromDataImpl(allocator, layout, storage, 0);
}

pub fn fromData(allocator: std.mem.Allocator, comptime dtype_i: DataType, shapes_a: []const usize, data: std.ArrayList(dtype_i.toTypeComp())) anyerror!Self {
    const buf_r: [*]u8 = @ptrCast(data.items.ptr);
    const bytes_size = dtype_i.dtypeSize() * data.items.len;

    return Self.fromDataRaw(allocator, dtype_i, shapes_a, Storage.Device.Cpu, buf_r, bytes_size);
}

pub fn fromSlice(allocator: std.mem.Allocator, comptime dtype_i: DataType, shapes_a: []const usize, data: []const dtype_i.toTypeComp()) anyerror!Self {
    var data_list = try std.ArrayList(dtype_i.toTypeComp()).initCapacity(allocator, data.len);
    try data_list.appendSlice(allocator, data);

    return try Self.fromData(allocator, dtype_i, shapes_a, data_list);
}

pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
    const layout = try self.layout.clone();

    if (!self.layout.isContiguous()) {
        return try self.contiguous();
    } else {
        const storage = try self.storage.deepCopy();

        return try Self.fromDataImpl(allocator, layout, storage, self._storage_offset);
    }
}

pub fn fromShapedData(allocator: std.mem.Allocator, comptime arr: anytype) anyerror!Self {
    const T = utils.getArrayRefItemType(@TypeOf(arr));
    const dtype_i = comptime DataType.typeToDataType(T);

    const shapes_i = utils.getArrayRefShapes(@TypeOf(arr));

    const buf_r: []u8 = @ptrCast(@constCast(arr));

    const bytes_size = buf_r.len;

    return Self.fromDataRaw(allocator, dtype_i, shapes_i, Storage.Device.Cpu, buf_r.ptr, bytes_size);
}

pub fn contiguous(self: *const Self) !Self {
    // if (self.layout.isContiguous()) {
    //     return self.dupe();
    // }

    const elem_size = self.dtype().dtypeSize();

    const new_buf = try self.allocator.alloc(u8, self.byteSize() * elem_size);

    var idx: usize = 0;
    const indices = try self.allocator.alloc(usize, self.ndim());

    const inner_scope = struct {
        fn copyRecursive(tensor: *const Self, indices_i: []usize, dim: usize, new_buf_i: []u8, idx_i: *usize, elem_size_a: usize) void {
            if (dim == tensor.ndim()) {
                var offset: usize = tensor._storage_offset;
                for (indices_i, 0..) |ind, i| {
                    offset += ind * tensor.strides()[i];
                }
                offset *= elem_size_a;

                const src = tensor.rawDataSlice()[offset .. offset + elem_size_a];
                const dst = new_buf_i[idx_i.* * elem_size_a .. (idx_i.* + 1) * elem_size_a];
                @memcpy(dst, src);

                idx_i.* += 1;
                return;
            }

            const shape_dim = tensor.shapes()[dim];
            for (0..shape_dim) |i| {
                indices_i[dim] = i;
                copyRecursive(tensor, indices_i, dim + 1, new_buf_i, idx_i, elem_size_a);
            }
        }
    };

    inner_scope.copyRecursive(self, indices, 0, new_buf, &idx, elem_size);

    var shapes_a = try std.ArrayList(usize).initCapacity(self.allocator, self.ndim());
    try shapes_a.appendSlice(self.allocator, self.shapes());

    return Self.fromDataRaw(self.allocator, self.layout.dtype(), shapes_a.items, Storage.Device.Cpu, @as([*]u8, @ptrCast(new_buf.ptr)), self.storage.byteSize());
}

pub fn arange(allocator: std.mem.Allocator, comptime data_type: DataType, args: struct { start: data_type.toTypeComp() = @as(data_type.toTypeComp(), 0), end: data_type.toTypeComp(), step: data_type.toTypeComp() = @as(data_type.toTypeComp(), 1) }) !Self {
    const T = data_type.toTypeComp();

    var arr_list = try std.ArrayList(T).initCapacity(allocator, 10);

    const start = args.start;
    const step = args.step;

    var tmp = start;

    while (tmp < args.end) {
        try arr_list.append(allocator, tmp);
        tmp += step;
    }

    var shape_list = try std.ArrayList(usize).initCapacity(allocator, 1);
    try shape_list.append(allocator, arr_list.items.len);

    return Self.fromData(allocator, data_type, shape_list.items, arr_list);
}

pub fn rand(allocator: std.mem.Allocator, shapes_a: []const usize, low: f32, high: f32) !Self {
    const total = utils.product(shapes_a);

    const buf = try allocator.alloc(f32, total);

    var rpng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = rpng.random();

    for (buf) |*x| {
        const u = rng.float(f32);
        x.* = low + (high - low) * u;
    }

    return Self{
        .allocator = allocator,
        .storage = Storage.init(allocator, Storage.Device.Cpu, @as([*]u8, @ptrCast(buf.ptr)), total * @sizeOf(f32)),
        .layout = try Layout.init(allocator, DataType.f32, shapes_a),
    };
}

pub fn randNorm(allocator: std.mem.Allocator, shapes_a: []const usize, mean_a: f32, stddev: f32) !Self {
    const total = utils.product(shapes_a);

    const buf = try allocator.alloc(f32, total);

    var rpng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rng = rpng.random();

    for (buf) |*x| {
        const u = rng.floatNorm(f32);
        x.* = mean_a + stddev * u;
    }

    return Self{
        .allocator = allocator,
        .storage = Storage.init(allocator, Storage.Device.Cpu, @as([*]u8, @ptrCast(buf.ptr)), total * @sizeOf(f32)),
        .layout = try Layout.init(allocator, DataType.f32, shapes_a),
    };
}

// attributes
fn dataSlice(self: *const Self, comptime T: anytype) []T {
    return self.storage.dataSlice(T);
}

pub fn rawDataSlice(self: *const Self) []u8 {
    return self.storage.rawDataSlice()[0..self.storage.byteSize()];
}

pub fn getWithIndices(self: *const Self, indices: []const usize) !Scalar {
    var idx = try utils.indices_to_flat(indices, self.shapes(), self.strides());
    idx += self._storage_offset;

    return switch (self.dtype()) {
        inline else => |v| Scalar.from(self.dataSlice(v.toTypeComp())[idx]),
    };
}

pub fn broadcast_to(self: *const Self, target_shape: []const usize) !Self {
    const new_strides = try utils.broadcastShapes(self.allocator, self.shapes(), self.strides(), target_shape);

    var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, target_shape.len);
    try new_shapes.appendSlice(self.allocator, target_shape);

    const layout = try Layout.initRaw(self.allocator, self.dtype(), new_shapes, new_strides);
    const storage = self.storage.clone();

    return try Self.fromDataImpl(self.allocator, layout, storage, self._storage_offset);
}

pub fn broadcast_to_(self: *Self, target_shape: []const usize) !void {
    const new_strides = try utils.broadcastShapes(self.allocator, self.shapes(), self.strides(), target_shape);

    var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, target_shape.len);
    try new_shapes.appendSlice(self.allocator, target_shape);

    const layout = try Layout.initRaw(self.allocator, self.dtype(), new_shapes, new_strides);

    self.layout = layout;
}

pub fn getWithIndicesCompType(self: *const Self, comptime data_type: DataType, indices: []const usize) !*data_type.toTypeComp() {
    var idx = try utils.indices_to_flat(indices, self.shapes(), self.strides());
    idx += self._storage_offset;

    return &self.dataSlice(data_type.toTypeComp())[idx];
}

// data transform
// pub fn map_inplace(self: *Self, dtype: DataType, f: fn (dtype.toType()) dtype.toType()) void {
//     for (self.data_slice(dtype)) |*val| {
//         val.* = f(val.*);
//     }
// }

//  pub fn map(self: *const Self, allocator: std.mem.Allocator, comptime dtype1: DataType, f: fn (T) dtype1.toType()) anyerror!Tensor(dtype1) {
//      const data = if (self.data) |d| d else return error.DataIsNull;

//      var new_data = try std.ArrayList(dtype1.toType()).initCapacity(allocator, data.items.len);
//      try new_data.appendNTimes(allocator, undefined, data.items.len);
//      for (data.items, 0..) |val, i| {
//          new_data.items[i] = f(val);
//      }

//      const CopyFn = struct {
//          pub inline fn copy(allocator_i: std.mem.Allocator, is: std.ArrayList(usize)) anyerror!InnerSlice {
//              if (is_static_rank) {
//                  return is;
//              } else {
//                  return try is.clone(allocator_i);
//              }
//          }
//      };

//      const NewTensorTyp = Tensor(dtype1, DimsTmpl);
//      return NewTensorTyp{ ._shape = try CopyFn.copy(allocator, self._shape), ._strides = try CopyFn.copy(allocator, self._strides), .allocator = allocator, .data = new_data };
//  }

// change shape
pub fn transpose(self: *const Self) anyerror!Self {
    const new_layout = try self.layout.transpose(0, 1);
    const new_storage = self.storage.clone();

    return Self{
        .allocator = self.allocator,
        .storage = new_storage,
        .layout = new_layout,
    };
}

pub fn permute(self: *const Self, perm: []const usize) anyerror!Self {
    const obj = if (!self.layout.isContiguous()) &(try self.contiguous()) else self;

    const new_layout = try obj.layout.permute(perm);
    const new_storage = obj.storage.clone();

    return Self{
        .allocator = obj.allocator,
        .storage = new_storage,
        .layout = new_layout,
    };
}

pub fn reshape(self: *const Self, new_shapes: []const usize) anyerror!Self {
    const obj = if (!self.layout.isContiguous()) &(try self.contiguous()) else self;

    const new_layout = try obj.layout.reshape(new_shapes);
    const new_storage = obj.storage.clone();

    return Self{
        .allocator = obj.allocator,
        .storage = new_storage,
        .layout = new_layout,
    };
}

pub fn unsqueeze(self: *const Self, dim: usize) anyerror!Self {
    var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, self.byteSize());
    try new_shapes.appendSlice(self.allocator, self.layout.shapes());
    try new_shapes.insert(self.allocator, dim, 1);

    return try self.reshape(new_shapes.items);
}

pub fn squeeze(self: *const Self, dim: ?usize) anyerror!Self {
    var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, self.byteSize());

    if (dim) |d| {
        for (self.shapes(), 0..) |s, i| {
            if (i == d and s == 1) continue;

            try new_shapes.append(self.allocator, s);
        }
    } else {
        for (self.shapes()) |s| {
            if (s == 1) {
                continue;
            }

            try new_shapes.append(self.allocator, s);
        }
    }

    return try self.reshape(new_shapes.items);
}

// core method
pub fn deinit(self: *const Self) void {
    self.storage.release();
}

fn getDataWithIdx(self: *const Self, dtype_i: DataType, idx: usize) Scalar {
    const c_s = @constCast(self);
    return switch (dtype_i) {
        .f16 => Scalar{ .f16 = c_s.dataSlice(DataType.f16.toType())[idx] },
        .f32 => Scalar{ .f32 = c_s.dataSlice(DataType.f32.toType())[idx] },
        .i32 => Scalar{ .i32 = c_s.dataSlice(DataType.i32.toType())[idx] },
        .u32 => Scalar{ .u32 = c_s.dataSlice(DataType.u32.toType())[idx] },
    };
}

pub fn byteSize(self: *const Self) usize {
    return self.layout.size() * self.layout.dtypeSize();
}

pub fn dtype(self: *const Self) DataType {
    return self.layout.dtype();
}

pub fn ndim(self: *const Self) usize {
    return self.layout.ndim();
}

pub fn shapes(self: *const Self) []const usize {
    return self.layout.shapes();
}

pub fn strides(self: *const Self) []const usize {
    return self.layout.strides();
}

pub fn equal(self: *const Self, other: *const Self) bool {
    if (!self.layout.equal(other.layout)) return false;

    const self_data_slice = self.dataSlice(self.layout.dtype());
    const other_data_slice = other.dataSlice(other.layout.dtype());

    return std.mem.eql(self.layout.dtype(), self_data_slice, other_data_slice);
}

pub fn approxEqual(self: *const Self, other: *const Self, comptime dtype_i: DataType, relEps: dtype_i.toType(), absEps: dtype_i.toType()) bool {
    if (!self.layout.equal(&other.layout)) return false;

    if (self.dtype() != other.dtype()) return false;

    if (self.dtype() != dtype_i) return false;

    const self_data_slice = self.dataSlice(dtype_i.toType());
    const other_data_slice = other.dataSlice(dtype_i.toType());
    return utils.sliceApproxEqual(dtype_i.toType(), self_data_slice, other_data_slice, relEps, absEps);
}

pub fn format(
    self: @This(),
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.print(
        \\Tensor{{
        \\.{f}
        \\.Data =
    , .{self.layout});

    _ = try writer.write("\n");

    self.fmtRecursive(writer, 0, &.{}) catch |err| {
        std.debug.print("meet failure: {}", .{err});
        return std.Io.Writer.Error.WriteFailed;
    };
    _ = try writer.write("\n}");
}

fn fmtRecursive(self: *const Self, writer: *std.Io.Writer, depth: usize, indices: []const usize) anyerror!void {
    const dims = self.ndim();

    if (depth == dims) {
        try writer.print("{f}", .{try self.getWithIndices(indices)});
    } else if (depth == dims - 1) {
        try self.fmt1dSlice(writer, indices);
    } else {
        try self.fmtNdSlice(writer, depth, indices);
    }
}

fn fmtNdSlice(self: *const Self, writer: *std.Io.Writer, depth: usize, base_indices: []const usize) anyerror!void {
    const pad_show_count = 4;

    const current_dim_size = self.shapes()[depth];
    const dims = self.ndim();

    _ = try writer.write("[");

    const show_all = current_dim_size <= 2 * pad_show_count;

    var slice_indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
    defer slice_indices.deinit(self.allocator);

    if (show_all) {
        for (0..current_dim_size) |i| {
            try slice_indices.append(self.allocator, i);
        }
    } else {
        for (0..pad_show_count) |i| {
            try slice_indices.append(self.allocator, i);
        }

        for (current_dim_size - pad_show_count..current_dim_size) |i| {
            try slice_indices.append(self.allocator, i);
        }
    }

    for (slice_indices.items, 0..) |slice_idx, idx| {
        if (idx > 0) {
            if (depth == dims - 2) {
                _ = try writer.write("\n ");
            } else {
                _ = try writer.write("\n\n ");
            }

            for (0..depth) |_| {
                _ = try writer.write(" ");
            }
        }

        if (!show_all and idx == pad_show_count) {
            if (depth == dims - 2) {
                _ = try writer.write("\n ");

                for (0..depth) |_| {
                    _ = try writer.write(" ");
                }

                _ = try writer.write("...\n ");
            } else {
                _ = try writer.write("\n ...\n\n ");
            }

            for (0..depth) |_| {
                _ = try writer.write(" ");
            }
        }

        var indices = try std.ArrayList(usize).initCapacity(self.allocator, 4);
        defer indices.deinit(self.allocator);

        try indices.appendSlice(self.allocator, base_indices);
        try indices.append(self.allocator, slice_idx);

        try self.fmtRecursive(writer, depth + 1, indices.items);
    }

    _ = try writer.write("]");
}

fn fmt1dSlice(self: *const Self, writer: *std.Io.Writer, base_indices: []const usize) anyerror!void {
    const pad_show_count = 5;

    const max_items: usize = if (base_indices.len == 0) 10 else 2 * pad_show_count;
    const current_dim_size = self.shapes()[base_indices.len];

    const line_size = 18;

    _ = try writer.write("[");

    const allocator = self.allocator;
    if (current_dim_size <= max_items) {
        for (0..current_dim_size) |i| {
            if (i > 0) {
                if (i % line_size == 0) {
                    _ = try writer.write("\n");
                } else {
                    _ = try writer.write(" ");
                }
            }

            var indices = try std.ArrayList(usize).initCapacity(allocator, 4);
            defer indices.deinit(allocator);

            try indices.appendSlice(allocator, base_indices);
            try indices.append(allocator, i);

            try writer.print("{f}", .{try self.getWithIndices(indices.items)});
        }
    } else {
        for (0..pad_show_count) |i| {
            if (i > 0) {
                _ = try writer.write(" ");
            }

            var indices = try std.ArrayList(usize).initCapacity(allocator, 4);
            defer indices.deinit(allocator);

            try indices.appendSlice(allocator, base_indices);
            try indices.append(allocator, i);

            try writer.print("{f}", .{try self.getWithIndices(indices.items)});
        }
        _ = try writer.write(" ... ");

        for (current_dim_size - pad_show_count..current_dim_size) |i| {
            var indices = try std.ArrayList(usize).initCapacity(allocator, 4);
            defer indices.deinit(allocator);

            try indices.appendSlice(allocator, base_indices);
            try indices.append(allocator, i);

            try writer.print("{f}", .{try self.getWithIndices(indices.items)});

            if (i < current_dim_size - 1) {
                _ = try writer.write(" ");
            }
        }
    }

    _ = try writer.write("]");
}

fn getArrayRefBuf(comptime arr: anytype) struct { [*]u8, usize } {
    const info = @typeInfo(@TypeOf(arr));

    const buf: []u8 = switch (info) {
        .pointer => |ptr| switch (ptr.size) {
            .one => switch (@typeInfo(ptr.child)) {
                .array => @as([]u8, @ptrCast(@constCast(arr)))[0..],
                .pointer => |pp| switch (pp.size) {
                    .slice => arr.*,
                    else => @compileError("only support pointer to one"),
                },
                else => @compileError(std.fmt.comptimePrint("only support pointer to one: data_type= {any}", .{info})),
            },
            else => @compileError("only support pointer to one"),
        },
        else => @compileError("only support pointer to one"),
    };

    return .{ buf.ptr, buf.len };
}

test "dyn tensor create" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const arr1 = [3][2]f32{
        [2]f32{ 1.0, 2.0 },
        [2]f32{ 3.0, 4.0 },
        [2]f32{ 5.0, 6.0 },
    };
    const t111 = try Self.fromShapedData(allocator, &arr1);

    const arr2 = [2][4]f32{
        [4]f32{ 3.0, 4.0, 5.0, 6.0 },
        [4]f32{ 5.0, 6.0, 7.0, 8.0 },
    };
    const t112 = try Self.fromShapedData(allocator, &arr2);

    const res_arr = [3][4]f32{
        [4]f32{ 13.0, 16.0, 19.0, 22.0 },
        [4]f32{ 29.0, 36.0, 43.0, 50.0 },
        [4]f32{ 45.0, 56.0, 67.0, 78.0 },
    };
    const res_t11 = try Self.fromShapedData(allocator, &res_arr);

    const t113 = try t111.matmul(&t112);
    std.debug.print("t111: {f} t112: {f}\n", .{ t111, t112 });
    std.debug.print("t113: {f}\n", .{t113});

    const compare_res = res_t11.approxEqual(&t113, DataType.f32, 0.001, 0.0001);
    try std.testing.expect(compare_res);
}

test "shape test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const arr1 = [3][2]f32{
        [2]f32{ 1.0, 2.0 },
        [2]f32{ 3.0, 4.0 },
        [2]f32{ 5.0, 6.0 },
    };
    const t111 = try Self.fromShapedData(allocator, &arr1);

    const t111_transposed = try t111.transpose();

    try std.testing.expect(t111.shapes()[0] == t111_transposed.shapes()[1]);
    try std.testing.expect(t111.shapes()[1] == t111_transposed.shapes()[0]);
    try std.testing.expectEqual(t111.getWithIndices(&.{ 0, 1 }), t111_transposed.getWithIndices(&.{ 1, 0 }));

    std.debug.print("t111: {f} t111 transposed: {f}\n", .{ t111, t111_transposed });

    // const arr2 = [2][4]f32{
    //
    //     [4]f32{ 3.0, 4.0, 5.0, 6.0 },
    //     [4]f32{ 5.0, 6.0, 7.0, 8.0 },
    // };
    // const t112 = try Tensor.fromShapedData(allocator, &arr2);
    const t111_unsqueezed = try t111.unsqueeze(1);
    try std.testing.expectEqualSlices(usize, t111_unsqueezed.shapes(), &.{ 3, 1, 2 });
    const t111_squeezed = try t111_unsqueezed.squeeze(null);
    try std.testing.expectEqualSlices(usize, t111_squeezed.shapes(), &.{ 3, 2 });

    std.debug.print("unsqueezed: {f} squeezed: {f}\n", .{ t111_unsqueezed, t111_squeezed });
}

test "random test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t1 = try Self.rand(allocator, &.{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t1: {f}\n", .{t1});

    const t2 = try Self.randNorm(allocator, &.{ 3000, 3000 }, 0.0, 1.0);
    std.debug.print("t2: {f}\n", .{t2});

    const t2_tc = try (try t2.transpose()).contiguous();

    const begin = std.time.milliTimestamp();
    const t3 = try t1.matmul(&t2_tc);
    const end = std.time.milliTimestamp();

    std.debug.print("t3: {f}\nelapsed: {d} microseconds\n", .{ t3, end - begin });
}

test "permute test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t1 = try Self.rand(allocator, &.{ 1, 2, 3, 4, 5, 6 }, 0.0, 2.0);
    const t1p = try t1.permute(&.{ 5, 4, 3, 2, 1, 0 });

    std.debug.print("t1: {f}\nt1p: {f}\n", .{ t1.layout, t1p.layout });
}

test "contiguous test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t1 = try Self.fromSlice(allocator, DataType.f32, &.{ 3, 4 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
    std.debug.print("t1: {f}\n", .{t1});

    const t1_ds = t1.dataSlice(f32);

    std.debug.print("t1 ds: {any}\n", .{t1_ds});

    const t1t = try t1.transpose();
    std.debug.print("t1t: {f}\n", .{t1t});

    const t1tc = try t1t.contiguous();
    std.debug.print("t1tc: {f}\n", .{t1tc});
    try std.testing.expect(t1tc.layout.isContiguous());

    std.debug.print("t1t ds: {any}\nt1tc ds: {any}\n", .{ t1t.dataSlice(f32), t1tc.dataSlice(f32) });

    try std.testing.expectApproxEqAbs((try t1t.getWithIndicesCompType(DataType.f32, &.{ 0, 2 })).*, (try t1tc.getWithIndicesCompType(DataType.f32, &.{ 0, 2 })).*, 0.00001);

    // var shape_1 = try std.ArrayList(usize).initCapacity(allocator, 10);
    // try shape_1.appendSlice(allocator, &.{3, 4});

    // const data = try std.ArrayList(f32).initCapacity(allocator, num: usize)
    // const t1 = try Tensor.fromData(allocator: Allocator, comptime dtype_i: DataType, shapes_a: Aligned(usize), data: Aligned(either type))
}

test "cat stack" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t1 = try Self.rand(allocator, &.{ 1, 2, 2 }, 0.0, 2.0);

    const t2 = try Self.rand(allocator, &.{ 1, 2, 2 }, 0.0, 2.0);

    const t3 = try Self.cat(allocator, &.{ t1, t2 }, 2);

    const t4 = try Self.stack(allocator, &.{ t1, t2 }, 2);
    std.debug.print("t1: {f} t2: {f} t3: {f} t4: {f}\n", .{ t1.layout, t2.layout, t3.layout, t4.layout });
}

test "split unbind" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t = try Self.arange(allocator, DataType.f32, .{ .end = 20 });
    std.debug.print("t: {f}\n", .{t});

    const t1 = try t.reshape(&.{ 2, 2, 5 });

    {
        const results = try t1.split(2, 2);

        std.debug.print("t1: {f}\n", .{t1});
        for (results) |result| {
            std.debug.print("result: {f} offset= {}\n", .{ result, result._storage_offset });
        }
    }

    std.debug.print("begin chunk\n", .{});
    {
        const results = try t1.chunk(5, 2);

        std.debug.print("t1: {f}\n", .{t1});
        for (results) |result| {
            std.debug.print("result: {f} offset= {}\n", .{ result, result._storage_offset });
        }
    }

    std.debug.print("begin unbind\n", .{});
    {
        const results = try t1.unbind(2);

        std.debug.print("t1: {f}\n", .{t1});
        for (results) |result| {
            std.debug.print("result: {f} offset= {} contiguoused= {f}\n", .{ result, result._storage_offset, try result.contiguous() });
        }
    }
}

test "map" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    var t = try Self.arange(allocator, DataType.f32, .{ .end = 10 });

    const func = struct {
        fn call(x: f32) f32 {
            return x * 3;
        }
    }.call;
    try t.map_(DataType.f32, func);
    std.debug.print("t: {f}\n", .{t});

    try t.add_(11.0);
    try t.add_(t);
    std.debug.print("add t: {f}\n", .{t});

    try t.mul_(2.0);
    std.debug.print("mul t: {f}\n", .{t});

    try t.sin_();
    try t.exp_();

    std.debug.print("t: {f}\n", .{t});

    try t.clamp_(0.0, 2.39);
    std.debug.print("t: {f}\n", .{t});

    const t1 = try t.add(t);
    std.debug.print("t: {f} t1: {f}\n", .{ t, t1 });
}

test "reduce" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    var a1 = try std.ArrayList(usize).initCapacity(allocator, 10);
    try a1.appendNTimes(allocator, 10, 10);
    std.debug.print("a1: {any}\n", .{a1});
    _ = a1.orderedRemove(0);

    std.debug.print("a1: {any}\n", .{a1});

    {
        const t1 = try Self.arange(allocator, DataType.f32, .{ .end = 10 });
        const t2 = try t1.sum(DataType.f32, 0);
        const t3 = try t1.mean(DataType.f32, 0);
        std.debug.print("t1: {f} t2: {f} t2 item: {f} t3: {f}\n", .{ t1, t2, try t2.scalarItem(), t3 });
    }

    {
        const t1 = try (try Self.arange(allocator, DataType.f32, .{ .end = 10 })).reshape(&.{ 2, 5 });
        const t2 = try t1.sum(DataType.f32, 1);
        const t3 = try t1.mean(DataType.f32, 1);
        std.debug.print("t1: {f} t2: {f} t3: {f}\n", .{ t1, t2, t3 });
    }
}

test "binary op" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t = try Self.arange(allocator, DataType.f32, .{ .end = 10 });
    std.debug.print("typ: {any}\n", .{@typeInfo(@TypeOf(&t))});
}

test "iterator" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t = try Self.arange(allocator, DataType.f32, .{ .end = 10 });
    const t1 = try t.reshape(&.{ 2, 5 });

    var iter1 = try t1.dataIter();
    while (iter1.next()) |item| {
        std.debug.print("item: {}\n", .{item});
    }
    // std.debug.print("typ: {any}\n", .{@typeInfo(@TypeOf(&t))});

}
