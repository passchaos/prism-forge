const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");

const dtype = @import("./dtype.zig");
const DataType = dtype.DataType;
const Scalar = dtype.Scalar;

const Layout = @import("./Layout.zig");

const Storage = @import("./Storage.zig");

const asSlice = utils.asSlice;

fn indices_to_flat(indices: []const usize, shape: []const usize, strides: []const usize) anyerror!usize {
    if (indices.len == 0) {
        return error.EmptyIndices;
    }

    var flat_index: usize = 0;
    for (indices, shape, 0..) |index, dim, idx| {
        if (index >= dim) {
            return error.OutOfBounds;
        }

        flat_index += index * strides[idx];
    }
    return flat_index;
}
fn flat_to_indices(flat_index: usize, strides: []const usize) []const usize {
    var indices = [_]usize{0} ** strides.len;
    for (0..strides.len) |dim| {
        indices[dim] = flat_index / strides[dim];
        flat_index %= strides[dim];
    }
    return indices;
}

pub const Tensor = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    storage: Storage,
    layout: Layout,

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

        const a = lhs.storage.dataSlice(f32);
        const b = rhs.storage.dataSlice(f32);

        const buf = try std.ArrayList(f32).initCapacity(lhs.allocator, m * n);
        const c = @as([*]f32, @ptrCast(buf.items.ptr));

        host.matmul(a, b, c, m, n, k);

        const data = @as([*]u8, @ptrCast(c));

        var shapes_i = try std.ArrayList(usize).initCapacity(lhs.allocator, 2);
        try shapes_i.appendSlice(lhs.allocator, &.{ m, n });
        return try Self.fromDataRaw(lhs.allocator, DataType.f32, shapes_i, Storage.Device.Cpu, m * n * @sizeOf(f32), data);
    }

    // create method
    pub fn fromDataRaw(allocator: std.mem.Allocator, dtype_i: DataType, shapes_a: std.ArrayList(usize), device: Storage.Device, bytes_size: usize, data: [*]u8) anyerror!Self {
        var expected_size: usize = 1;
        for (shapes_a.items) |shape| {
            expected_size *= shape;
        }
        expected_size *= dtype_i.dtypeSize();

        if (expected_size != bytes_size) {
            return error.InvalidSize;
        }

        const storage = Storage.init(allocator, device, data, bytes_size);

        const layout = try Layout.init(allocator, dtype_i, shapes_a);

        return Self{ .allocator = allocator, .storage = storage, .layout = layout };
    }

    pub fn fromData(allocator: std.mem.Allocator, comptime dtype_i: DataType, shapes_a: std.ArrayList(usize), data: std.ArrayList(dtype_i.toTypeComp())) anyerror!Self {
        const buf_r: [*]u8 = @ptrCast(data.items.ptr);
        const bytes_size = dtype_i.dtypeSize() * data.items.len;

        return Self.fromDataRaw(allocator, dtype_i, shapes_a, Storage.Device.Cpu, bytes_size, buf_r);
    }

    pub fn fromSlice(allocator: std.mem.Allocator, comptime dtype_i: DataType, shapes_a: []const usize, data: []const dtype_i.toTypeComp()) anyerror!Self {
        var shape_list = try std.ArrayList(usize).initCapacity(allocator, shapes_a.len);
        try shape_list.appendSlice(allocator, shapes_a);

        var data_list = try std.ArrayList(dtype_i.toTypeComp()).initCapacity(allocator, data.len);
        try data_list.appendSlice(allocator, data);

        return try Self.fromData(allocator, dtype_i, shape_list, data_list);
    }

    pub fn fromShapedData(allocator: std.mem.Allocator, comptime arr: anytype) anyerror!Self {
        const T = utils.getArrayRefItemType(@TypeOf(arr));
        const dtype_i = comptime DataType.typeToDataType(T);

        const shapes_i = utils.getArrayRefShapes(@TypeOf(arr));

        const buf_r: []u8 = @ptrCast(@constCast(arr));

        const bytes_size = buf_r.len;

        var arr_list = try std.ArrayList(usize).initCapacity(allocator, shapes_i.len);
        try arr_list.appendSlice(allocator, shapes_i);

        return Self.fromDataRaw(allocator, dtype_i, arr_list, Storage.Device.Cpu, bytes_size, buf_r.ptr);
    }

    pub fn contiguous(self: *const Self) !Self {
        // if (self.layout.isContiguous()) {
        //     return self.dupe();
        // }

        const elem_size = self.dtype().dtypeSize();

        const new_buf = try self.allocator.alloc(u8, self.size() * elem_size);

        var idx: usize = 0;
        const indices = try self.allocator.alloc(usize, self.ndim());

        const inner_scope = struct {
            fn copyRecursive(tensor: *const Self, indices_i: []usize, dim: usize, new_buf_i: []u8, idx_i: *usize, elem_size_a: usize) void {
                if (dim == tensor.ndim()) {
                    var offset: usize = 0;
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

        return Self.fromDataRaw(self.allocator, self.layout.dtype(), shapes_a, Storage.Device.Cpu, self.storage.byteSize(), @as([*]u8, @ptrCast(new_buf.ptr)));
    }

    pub fn rand(allocator: std.mem.Allocator, shapes_a: std.ArrayList(usize), low: f32, high: f32) !Self {
        const total = utils.product(shapes_a.items);

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

    pub fn randNorm(allocator: std.mem.Allocator, shapes_a: std.ArrayList(usize), mean: f32, stddev: f32) !Self {
        const total = utils.product(shapes_a.items);

        const buf = try allocator.alloc(f32, total);

        var rpng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const rng = rpng.random();

        for (buf) |*x| {
            const u = rng.floatNorm(f32);
            x.* = mean + stddev * u;
        }

        return Self{
            .allocator = allocator,
            .storage = Storage.init(allocator, Storage.Device.Cpu, @as([*]u8, @ptrCast(buf.ptr)), total * @sizeOf(f32)),
            .layout = try Layout.init(allocator, DataType.f32, shapes_a),
        };
    }

    // attributes
    pub fn dataSlice(self: *const Self, comptime T: anytype) []const T {
        return self.storage.dataSlice(T)[0..self.size()];
    }

    pub fn rawDataSlice(self: *const Self) []u8 {
        return self.storage.rawDataSlice()[0..self.storage.byteSize()];
    }

    pub fn getWithIndices(self: *const Self, comptime dtype_i: DataType, indices: []const usize) !dtype_i.toType() {
        const flat_index = try indices_to_flat(indices, self.shapes(), self.strides());
        return self.dataSlice(dtype_i.toType())[flat_index];
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
        const new_layout = try self.layout.transpose();
        const new_storage = self.storage.clone();

        return Self{
            .allocator = self.allocator,
            .storage = new_storage,
            .layout = new_layout,
        };
    }

    pub fn reshape(self: *const Self, new_shapes: std.ArrayList(usize)) anyerror!Self {
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
        var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, self.size());
        try new_shapes.appendSlice(self.allocator, self.layout.shapes());
        try new_shapes.insert(self.allocator, dim, 1);

        return try self.reshape(new_shapes);
    }

    pub fn squeeze(self: *const Self, dim: ?usize) anyerror!Self {
        var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, self.size());

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

        return try self.reshape(new_shapes);
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

    pub fn size(self: *const Self) usize {
        return self.storage.byteSize() / self.layout.dtypeSize();
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
            const flat_index = try indices_to_flat(indices, asSlice(&self.shapes()), asSlice(&self.strides()));

            try writer.print("{f}", .{self.getDataWithIdx(self.layout.dtype(), flat_index)});
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

                const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shapes()), asSlice(&self.strides()));

                try writer.print("{f}", .{self.getDataWithIdx(self.layout.dtype(), flat_idx)});
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

                const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shapes()), asSlice(&self.strides()));

                try writer.print("{f}", .{self.getDataWithIdx(self.layout.dtype(), flat_idx)});
            }
            _ = try writer.write(" ... ");

            for (current_dim_size - pad_show_count..current_dim_size) |i| {
                var indices = try std.ArrayList(usize).initCapacity(allocator, 4);
                defer indices.deinit(allocator);

                try indices.appendSlice(allocator, base_indices);
                try indices.append(allocator, i);

                const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shapes()), asSlice(&self.strides()));

                try writer.print("{f}", .{self.getDataWithIdx(self.layout.dtype(), flat_idx)});

                if (i < current_dim_size - 1) {
                    _ = try writer.write(" ");
                }
            }
        }

        _ = try writer.write("]");
    }
};

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
    const t111 = try Tensor.fromShapedData(allocator, &arr1);

    const arr2 = [2][4]f32{
        [4]f32{ 3.0, 4.0, 5.0, 6.0 },
        [4]f32{ 5.0, 6.0, 7.0, 8.0 },
    };
    const t112 = try Tensor.fromShapedData(allocator, &arr2);

    const res_arr = [3][4]f32{
        [4]f32{ 13.0, 16.0, 19.0, 22.0 },
        [4]f32{ 29.0, 36.0, 43.0, 50.0 },
        [4]f32{ 45.0, 56.0, 67.0, 78.0 },
    };
    const res_t11 = try Tensor.fromShapedData(allocator, &res_arr);

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
    const t111 = try Tensor.fromShapedData(allocator, &arr1);

    const t111_transposed = try t111.transpose();

    try std.testing.expect(t111.shapes()[0] == t111_transposed.shapes()[1]);
    try std.testing.expect(t111.shapes()[1] == t111_transposed.shapes()[0]);
    try std.testing.expectEqual(t111.getWithIndices(DataType.f32, &.{ 0, 1 }), t111_transposed.getWithIndices(DataType.f32, &.{ 1, 0 }));

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

    var shapes = try std.ArrayList(usize).initCapacity(allocator, 2);
    try shapes.appendSlice(allocator, &.{ 3000, 3000 });

    const t1 = try Tensor.rand(allocator, shapes, 0.0, 1.0);
    std.debug.print("t1: {f}\n", .{t1});

    const t2 = try Tensor.randNorm(allocator, shapes, 0.0, 1.0);
    std.debug.print("t2: {f}\n", .{t2});

    const begin = std.time.microTimestamp();
    const t3 = try t1.matmul(&(try t2.transpose()));
    const end = std.time.microTimestamp();

    std.debug.print("t3: {f}\nelapsed: {d} microseconds\n", .{ t3, end - begin });
}

test "contiguous test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const t1 = try Tensor.fromSlice(allocator, DataType.f32, &.{ 3, 4 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
    std.debug.print("t1: {f}\n", .{t1});

    const t1_ds = t1.dataSlice(f32);

    std.debug.print("t1 ds: {any}\n", .{t1_ds});

    const t1t = try t1.transpose();
    std.debug.print("t1t: {f}\n", .{t1t});

    const t1tc = try t1t.contiguous();
    std.debug.print("t1tc: {f}\n", .{t1tc});
    try std.testing.expect(t1tc.layout.isContiguous());

    std.debug.print("t1t ds: {any}\nt1tc ds: {any}\n", .{ t1t.dataSlice(f32), t1tc.dataSlice(f32) });

    try std.testing.expectApproxEqAbs(try t1t.getWithIndices(DataType.f32, &.{ 0, 2 }), try t1tc.getWithIndices(DataType.f32, &.{ 0, 2 }), 0.00001);

    // var shape_1 = try std.ArrayList(usize).initCapacity(allocator, 10);
    // try shape_1.appendSlice(allocator, &.{3, 4});

    // const data = try std.ArrayList(f32).initCapacity(allocator, num: usize)
    // const t1 = try Tensor.fromData(allocator: Allocator, comptime dtype_i: DataType, shapes_a: Aligned(usize), data: Aligned(either type))
}
