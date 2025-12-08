const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");

const asSlice = utils.asSlice;

pub const Device = enum { Cpu, Cuda };

pub const DataType = enum {
    f16,
    f32,
    i32,
    u32,

    pub fn typeToDataType(comptime T: type) DataType {
        return switch (T) {
            f16 => .f16,
            f32 => .f32,
            i32 => .i32,
            u32 => .u32,
            else => @compileError("Unsupported type: " ++ @typeName(T)),
        };
    }

    pub fn toType(self: DataType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
            .i32 => i32,
            .u32 => u32,
        };
    }

    pub fn toTypeComp(comptime self: DataType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
            .i32 => i32,
            .u32 => u32,
        };
    }
    pub fn dtypeSize(self: DataType) usize {
        return switch (self) {
            .f16 => 2,
            .f32 => 4,
            .i32 => 4,
            .u32 => 4,
        };
    }
};

const Scalar = union(enum) {
    f16: f16,
    f32: f32,
    i32: i32,
    u32: u32,

    pub fn format(self: @This(), writer: *std.io.Writer) std.Io.Writer.Error!void {
        return switch (self) {
            .f16 => writer.print("{d}", .{self.f16}),
            .f32 => writer.print("{d}", .{self.f32}),
            .i32 => writer.print("{d}", .{self.i32}),
            .u32 => writer.print("{d}", .{self.u32}),
        };
    }
};

fn product(arr: []const usize) usize {
    var result: usize = 1;
    for (arr) |item| {
        result *= item;
    }
    return result;
}

pub const Layout = struct {
    allocator: std.mem.Allocator,
    _dtype: DataType,
    _shapes: std.ArrayList(usize),
    _strides: std.ArrayList(usize),
    _transposed: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, comptime dt: DataType, shapes: std.ArrayList(usize)) !Self {
        const strides = try utils.computeStrides(allocator, shapes);

        const layout = Self{
            ._dtype = dt,
            ._shapes = shapes,
            ._strides = strides,
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

    pub fn size(self: *const Self) usize {
        return product(self._shapes.items);
    }

    pub fn equal(self: *const Self, other: *const Self) bool {
        return self._dtype == other._dtype and std.mem.eql(usize, self._shapes.items, other._shapes.items) and std.mem.eql(usize, self._strides.items, other._strides.items);
    }
};

pub const Storage = struct {
    allocator: std.mem.Allocator,
    device: Device,
    buf: [*]u8,
    bytes_size: usize,
    ref_count: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, device: Device, buf: [*]u8, bytes_size: usize) Self {
        return Self{
            .allocator = allocator,
            .device = device,
            .buf = buf,
            .bytes_size = bytes_size,
            .ref_count = 1,
        };
    }

    pub fn clone(self: *const Self) Self {
        const v_s = @constCast(self);
        v_s.retain();

        return Self{
            .allocator = self.allocator,
            .device = self.device,
            .buf = self.buf,
            .bytes_size = self.bytes_size,
            .ref_count = self.ref_count,
        };
    }

    pub fn deinit(self: *Self) void {
        self.release();
    }

    fn retain(self: *Self) void {
        self.ref_count += 1;
    }

    fn release(self: *Self) void {
        if (self.ref_count > 0) {
            self.ref_count -= 1;
        }

        if (self.ref_count == 0) {
            if (self.device == .Cpu) {
                if (self.buf) |buf| {
                    self.allocator.free(buf);
                }
            }
        }
    }
};

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

        if (self.shape()[1] != other.shape()[0]) {
            return error.ShapeMismatch;
        }

        const m = self.shape()[0];
        const n = other.shape()[1];
        const k = self.shape()[1];

        const a = @as([*]const f32, @ptrCast(@alignCast(self.storage.buf)));
        const b = @as([*]const f32, @ptrCast(@alignCast(other.storage.buf)));

        const buf = try std.ArrayList(f32).initCapacity(self.allocator, m * n);
        const c = @as([*]f32, @ptrCast(buf.items.ptr));

        host.matmul(a, b, c, m, n, k);

        const data = @as([*]u8, @ptrCast(c));

        var shapes = try std.ArrayList(usize).initCapacity(self.allocator, 2);
        try shapes.appendSlice(self.allocator, &.{ m, n });
        return try Self.fromData(self.allocator, DataType.f32, shapes, m * n * @sizeOf(f32), data);
    }

    // create method
    pub fn fromData(allocator: std.mem.Allocator, comptime dtype_i: DataType, shapes: std.ArrayList(usize), bytes_size: usize, data: [*]u8) anyerror!Self {
        var storage = Storage.init(allocator, Device.Cpu, data, bytes_size);
        storage.retain();

        const layout = try Layout.init(allocator, dtype_i, shapes);

        return Self{ .allocator = allocator, .storage = storage, .layout = layout };
    }

    pub fn fromShapedData(allocator: std.mem.Allocator, comptime arr: anytype) anyerror!Self {
        const T = utils.getArrayRefItemType(@TypeOf(arr));
        const dtype_i = comptime DataType.typeToDataType(T);

        const shapes = utils.getArrayRefShapes(@TypeOf(arr));

        const buf_r: []u8 = @ptrCast(@constCast(arr));

        const bytes_size = buf_r.len;

        var arr_list = try std.ArrayList(usize).initCapacity(allocator, shapes.len);
        try arr_list.appendSlice(allocator, shapes);

        var storage = Storage.init(allocator, Device.Cpu, buf_r.ptr, bytes_size);
        storage.retain();

        const layout = try Layout.init(allocator, dtype_i, arr_list);

        return Self{
            .allocator = allocator,
            .storage = storage,
            .layout = layout,
        };
    }

    pub fn rand(allocator: std.mem.Allocator, shapes: std.ArrayList(usize), low: f32, high: f32) !Self {
        const total = product(shapes.items);

        const buf = try allocator.alloc(f32, total);

        var rpng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const rng = rpng.random();

        for (buf) |*x| {
            const u = rng.float(f32);
            x.* = low + (high - low) * u;
        }

        return Self{
            .allocator = allocator,
            .storage = Storage.init(allocator, Device.Cpu, @as([*]u8, @ptrCast(buf.ptr)), total * @sizeOf(f32)),
            .layout = try Layout.init(allocator, DataType.f32, shapes),
        };
    }

    pub fn randNorm(allocator: std.mem.Allocator, shapes: std.ArrayList(usize), mean: f32, stddev: f32) !Self {
        const total = product(shapes.items);

        const buf = try allocator.alloc(f32, total);

        var rpng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const rng = rpng.random();

        for (buf) |*x| {
            const u = rng.floatNorm(f32);
            x.* = mean + stddev * u;
        }

        return Self{
            .allocator = allocator,
            .storage = Storage.init(allocator, Device.Cpu, @as([*]u8, @ptrCast(buf.ptr)), total * @sizeOf(f32)),
            .layout = try Layout.init(allocator, DataType.f32, shapes),
        };
    }

    // attributes
    pub fn dataSlice(self: *const Self, comptime dtype_i: DataType) []const dtype_i.toType() {
        const T = dtype_i.toType();
        const d_buf: [*]T = @ptrCast(@alignCast(self.storage.buf));

        const d_len = self.storage.bytes_size / dtype_i.dtypeSize();
        return d_buf[0..d_len];
    }

    pub fn getWithIndices(self: *const Self, comptime dtype_i: DataType, indices: []const usize) !dtype_i.toType() {
        const flat_index = try indices_to_flat(indices, self.shape(), self.strides());
        return self.dataSlice(dtype_i)[flat_index];
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
        const new_layout = try self.layout.reshape(new_shapes);
        const new_storage = self.storage.clone();

        return Self{
            .allocator = self.allocator,
            .storage = new_storage,
            .layout = new_layout,
        };
    }

    pub fn unsqueeze(self: *const Self, dim: usize) anyerror!Self {
        var new_shapes = try self.layout._shapes.clone(self.allocator);
        try new_shapes.insert(self.allocator, dim, 1);

        return try self.reshape(new_shapes);
    }

    pub fn squeeze(self: *const Self, dim: ?usize) anyerror!Self {
        var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, self.size());

        if (dim) |d| {
            for (self.shape(), 0..) |s, i| {
                if (i == d and s == 1) continue;

                try new_shapes.append(self.allocator, s);
            }
        } else {
            for (self.shape()) |s| {
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
            .f16 => Scalar{ .f16 = c_s.dataSlice(DataType.f16)[idx] },
            .f32 => Scalar{ .f32 = c_s.dataSlice(DataType.f32)[idx] },
            .i32 => Scalar{ .i32 = c_s.dataSlice(DataType.i32)[idx] },
            .u32 => Scalar{ .u32 = c_s.dataSlice(DataType.u32)[idx] },
        };
    }

    pub fn size(self: *const Self) usize {
        return self.storage.bytes_size / self.layout._dtype.dtypeSize();
    }

    pub fn dtype(self: *const Self) DataType {
        return self.layout._dtype;
    }

    pub fn ndim(self: *const Self) usize {
        return self.layout._shapes.items.len;
    }

    pub fn shape(self: *const Self) []const usize {
        return self.layout._shapes.items;
    }

    pub fn strides(self: *const Self) []const usize {
        return self.layout._strides.items;
    }

    pub fn equal(self: *const Self, other: *const Self) bool {
        if (!self.layout.equal(other.layout)) return false;

        const self_data_slice = self.dataSlice(self.layout._dtype);
        const other_data_slice = other.dataSlice(other.layout._dtype);

        return std.mem.eql(self.layout._dtype, self_data_slice, other_data_slice);
    }

    pub fn approxEqual(self: *const Self, other: *const Self, comptime dtype_i: DataType, relEps: dtype_i.toType(), absEps: dtype_i.toType()) bool {
        if (!self.layout.equal(&other.layout)) return false;

        if (self.dtype() != other.dtype()) return false;

        if (self.dtype() != dtype_i) return false;

        const self_data_slice = self.dataSlice(dtype_i);
        const other_data_slice = other.dataSlice(dtype_i);
        return utils.sliceApproxEqual(dtype_i.toType(), self_data_slice, other_data_slice, relEps, absEps);
    }

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.print(
            \\Tensor{{
            \\  .TypeName = {s}
            \\  .Dtype = {any},
            \\  .Shape = ({any}, {any}),
            \\  .Strides = ({any}, {any}),
            \\  .DataLen = {}
            \\  .Data =
        , .{ @typeName(@TypeOf(self)), self.layout._dtype, @TypeOf(self.shape()), self.shape(), @TypeOf(self.strides()), self.strides(), self.size() });

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
            const flat_index = try indices_to_flat(indices, asSlice(&self.shape()), asSlice(&self.strides()));

            try writer.print("{f}", .{self.getDataWithIdx(self.layout._dtype, flat_index)});
        } else if (depth == dims - 1) {
            try self.fmt1dSlice(writer, indices);
        } else {
            try self.fmtNdSlice(writer, depth, indices);
        }
    }

    fn fmtNdSlice(self: *const Self, writer: *std.Io.Writer, depth: usize, base_indices: []const usize) anyerror!void {
        const pad_show_count = 4;

        const current_dim_size = self.shape()[depth];
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
        const current_dim_size = self.shape()[base_indices.len];

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

                const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));

                try writer.print("{f}", .{self.getDataWithIdx(self.layout._dtype, flat_idx)});
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

                const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));

                try writer.print("{f}", .{self.getDataWithIdx(self.layout._dtype, flat_idx)});
            }
            _ = try writer.write(" ... ");

            for (current_dim_size - pad_show_count..current_dim_size) |i| {
                var indices = try std.ArrayList(usize).initCapacity(allocator, 4);
                defer indices.deinit(allocator);

                try indices.appendSlice(allocator, base_indices);
                try indices.append(allocator, i);

                const flat_idx = try indices_to_flat(indices.items, asSlice(&self.shape()), asSlice(&self.strides()));

                try writer.print("{f}", .{self.getDataWithIdx(self.layout._dtype, flat_idx)});

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

    try std.testing.expect(t111.shape()[0] == t111_transposed.shape()[1]);
    try std.testing.expect(t111.shape()[1] == t111_transposed.shape()[0]);
    try std.testing.expectEqual(t111.getWithIndices(DataType.f32, &.{ 0, 1 }), t111_transposed.getWithIndices(DataType.f32, &.{ 1, 0 }));

    std.debug.print("t111: {f} t111 transposed: {f}\n", .{ t111, t111_transposed });

    // const arr2 = [2][4]f32{
    //
    //     [4]f32{ 3.0, 4.0, 5.0, 6.0 },
    //     [4]f32{ 5.0, 6.0, 7.0, 8.0 },
    // };
    // const t112 = try Tensor.fromShapedData(allocator, &arr2);
    const t111_unsqueezed = try t111.unsqueeze(1);
    try std.testing.expectEqualSlices(usize, t111_unsqueezed.shape(), &.{ 3, 1, 2 });
    const t111_squeezed = try t111_unsqueezed.squeeze(null);
    try std.testing.expectEqualSlices(usize, t111_squeezed.shape(), &.{ 3, 2 });

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

// test "construction test" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     const allocator = gpa.allocator();

//     const DType = DataType;
//     const TensorF32x3x2 = Tensor(DType.f32, &.{ 3, 2 });

//     const arr1 = [3][2]f32{
//         [2]f32{ 1.0, 2.0 },
//         [2]f32{ 3.0, 4.0 },
//         [2]f32{ 5.0, 6.0 },
//     };
//     const t11 = try TensorF32x3x2.from_shaped_data(allocator, &arr1);
//     defer t11.deinit(allocator);
//     std.debug.print("t11: {f}\n", .{t11});

//     const Tensor3U32_1 = Tensor(DType.u32, &.{ 3, null, 5 });
//     const t3_1 = try Tensor3U32_1.full(allocator, .{ .shape = .{ 3, 4, 5 } }, 21);
//     defer t3_1.deinit(allocator);
//     std.debug.print("t3_1: {f}\n", .{t3_1});

//     const TensorU32 = Tensor(DType.u32, null);

//     var shape4 = try std.ArrayList(usize).initCapacity(allocator, 4);
//     try shape4.appendSlice(allocator, &.{ 2, 3, 3, 1, 5 });

//     const t4 = try TensorU32.full(allocator, .{ .shape = try shape4.clone(allocator) }, 24);
//     defer t4.deinit(allocator);
//     std.debug.print("t4: {f}\n", .{t4});

//     const t5 = try TensorU32.full(allocator, .{ .shape = shape4 }, 24);
//     defer t5.deinit(allocator);
//     std.debug.print("t5: {f} {any}\n", .{ t5, t5._shape });

//     const Tensor2 = Tensor(DType.f32, &.{ null, null });
//     const t6 = try Tensor2.eye(allocator, 10);
//     defer t6.deinit(allocator);
//     std.debug.print("t6: {f}\n", .{t6});
// }

// test "arange" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     const allocator = gpa.allocator();

//     const Tensor3 = Tensor(DataType.u32, &.{null});

//     const t1 = try Tensor3.arange_count(allocator, 1, 2, 20);
//     defer t1.deinit(allocator);
//     std.debug.print("t1: {f}\n", .{t1});

//     const t2 = try Tensor3.arange_step(
//         allocator,
//         1,
//         40,
//         2,
//     );
//     defer t2.deinit(allocator);
//     std.debug.print("t2: {f}\n", .{t2});
// }

// test "shape transform" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     const allocator = gpa.allocator();

//     const Tensor112 = Tensor(DataType.u32, &.{ 2, 2 });
//     const Tensor122 = Tensor(DataType.u32, &.{ 3, 3 });

//     var arr1 = try std.ArrayList(u32).initCapacity(allocator, 4);
//     try arr1.appendSlice(allocator, &[_]u32{ 1, 2, 3, 4 });
//     var t112 = try Tensor112.from_data(allocator, .{}, arr1);
//     defer t112.deinit(allocator);

//     try t112.transpose();

//     var t112_reshaped = try t112.reshape(&.{ 4, 1 });
//     defer t112_reshaped.deinit(allocator);
//     std.debug.print("t112 reshaped: {f}\n", .{t112_reshaped});

//     const t112_comp_reshaped = try t112_reshaped.reshapeComp(&.{ 4, 1 });
//     defer t112_comp_reshaped.deinit(allocator);
//     std.debug.print("t112 comp reshaped: {f}\n", .{t112_comp_reshaped});

//     std.debug.print("t112: {f}\n", .{t112});

//     const Tensor41 = Tensor(DataType.u32, &.{ 4, 1 });

//     var arr1_normal = try std.ArrayList(u32).initCapacity(allocator, 4);
//     try arr1_normal.appendSlice(allocator, &[_]u32{ 6, 7, 8, 9 });
//     const t112_normal = try Tensor41.from_data(allocator, .{}, arr1_normal);
//     defer t112_normal.deinit(allocator);
//     std.debug.print("t112 normal: {f}\n", .{t112_normal});

//     if (@TypeOf(t112_reshaped) == @TypeOf(t112_normal)) {
//         std.debug.print("same type\n", .{});
//     } else {
//         std.debug.print("different type\n", .{});
//     }

//     // defer t112_reshaped.deinit(allocator);

//     var arr2 = try std.ArrayList(u32).initCapacity(allocator, 6);
//     try arr2.appendSlice(allocator, &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//     var t122 = try Tensor122.from_data(allocator, .{}, arr2);
//     defer t122.deinit(allocator);

//     std.debug.print("t122: {f}\n", .{t122});
//     try t122.transpose();
//     std.debug.print("transposed t122: {f}\n", .{t122});

//     const Tensor22 = Tensor(DataType.f32, &.{ null, 5 });
//     var t22 = try Tensor22.eye(allocator, 5);
//     defer t22.deinit(allocator);

//     std.debug.print("t22: {f}\n", .{t22});
//     try t22.transpose();
//     std.debug.print("t22 transpose: {f}\n", .{t22});

//     const Tensor32 = Tensor(DataType.f32, null);
//     const t32 = try Tensor32.eye(allocator, 5);
//     defer t32.deinit(allocator);

//     const arr = [4][5]f32{ [_]f32{ 1, 2, 3, 4, 5 }, [_]f32{ 6, 7, 8, 9, 10 }, [_]f32{ 11, 12, 13, 14, 15 }, [_]f32{ 16, 17, 18, 19, 20 } };
//     var t312 = try Tensor32.from_shaped_data(allocator, &arr);
//     defer t312.deinit(allocator);

//     std.debug.print("t312: {f}\n", .{t312});

//     try t312.transpose();

//     std.debug.print("t312 transpose: {f}\n", .{t312});
// }

// test "map related methods" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();
//     const allocator = arena.allocator();
//     // const allocator = gpa.allocator();

//     const TensorF32x3x2 = Tensor(DataType.f32, &.{ 3, 2 });

//     const arr1 = [3][2]f32{
//         [2]f32{ 1.1, 2.2 },
//         [2]f32{ 3.3, 4.01 },
//         [2]f32{ 5.9, 6.1 },
//     };
//     var t11 = try TensorF32x3x2.from_shaped_data(allocator, &arr1);
//     std.debug.print("t11: {f}\n", .{t11});

//     const FnWithCtx = struct {
//         pub fn double(x: f32) f32 {
//             return 2.0 * x;
//         }
//         pub fn call(x: f32) i32 {
//             return @as(i32, @intFromFloat(x));
//         }
//     };

//     const t11_1 = try t11.map(allocator, DataType.i32, FnWithCtx.call);
//     std.debug.print("t11_1: {f}\n", .{t11_1});

//     t11.map_i(FnWithCtx.double);
//     std.debug.print("t11: {f}\n", .{t11});
// }

// test "equal judge" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();

//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const TensorF32x3x2 = Tensor(DataType.f32, &.{ 3, 2 });

//     const arr1 = [3][2]f32{
//         [2]f32{ 1.1, 2.2 },
//         [2]f32{ 3.3, 4.01 },
//         [2]f32{ 5.9, 6.1 },
//     };

//     const t11 = try TensorF32x3x2.from_shaped_data(allocator, &arr1);

//     const arr2 = [3][2]f32{
//         [2]f32{ 1.1, 2.2 },
//         [2]f32{ 3.3, 4.01 },
//         [2]f32{ 5.9, 6.1000005 },
//     };

//     const a: []const f32 = &.{ 5.9, 6.1000001 };
//     const b: []const f32 = &.{ 5.9, 6.1 };

//     try std.testing.expect(utils.sliceEqual(f32, a, b));

//     const t11_1 = try TensorF32x3x2.from_shaped_data(allocator, &arr2);
//     try std.testing.expect(!t11.equal(&t11_1));
//     try std.testing.expect(t11.approxEqual(&t11_1, 0.0000001, 0.00001));
// }
