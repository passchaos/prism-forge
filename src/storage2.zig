const std = @import("std");
const utils = @import("utils.zig");
const dtype = @import("dtype2.zig");

const DType = dtype.DType;

const Counter = struct {
    id: i32 = 0,

    pub fn next(self: *Counter) i32 {
        const current = self.id;
        self.id += 1;
        return current;
    }
};

const NextId = struct {
    var id: i32 = 0;

    /// adds one to the number given
    // adds two
    pub fn call() i32 {
        const current = id;
        id += 1;
        return current;
    }

    test call {
        const id1 = NextId.call();
        const id2 = NextId.call();
        try std.testing.expectEqual(id1, 0);
        try std.testing.expectEqual(id2, 1);
    }
};

test "counter_static" {
    var counter = Counter{};
    _ = counter.next();
    _ = counter.next();
    _ = counter.next();
    try std.testing.expectEqual(counter.id, 3);

    std.debug.print("next id: {} {}\n", .{ NextId.call(), NextId.call() });
}

const AllocRefCount = struct {
    allocator: std.mem.Allocator,
    count: std.atomic.Value(usize),

    const Self = @This();

    pub fn refCount(self: *const Self) usize {
        return self.count.load(.seq_cst);
    }
};

pub const RawBuffer = struct {
    dtype: DType,
    ptr: [*]u8,
    len: usize,
    alloc_ref_count: ?*AllocRefCount,

    const Self = @This();

    pub fn deinit(self: *const Self) void {
        self.release();
    }

    pub fn asSlice(self: *const Self) []u8 {
        return self.ptr[0..self.len];
    }

    pub fn dataCount(self: *const Self) usize {
        return self.len / self.dtype.dataLen();
    }

    pub fn fromBorrowed(ptr: [*]u8, len: usize, dtype_a: DType) Self {
        return Self{
            .dtype = dtype_a,
            .ptr = ptr,
            .len = len,
            .alloc_ref_count = null,
        };
    }

    pub fn fromOwned(allocator: std.mem.Allocator, ptr: [*]u8, len: usize, dtype_a: DType) !Self {
        var alloc_rc = try allocator.create(AllocRefCount);
        alloc_rc.allocator = allocator;
        alloc_rc.count = std.atomic.Value(usize).init(1);

        return Self{
            .dtype = dtype_a,
            .ptr = ptr,
            .len = len,
            .alloc_ref_count = alloc_rc,
        };
    }

    pub fn retain(self: *const Self) void {
        if (self.alloc_ref_count) |alloc_rc| {
            _ = alloc_rc.count.fetchAdd(1, .seq_cst);
        }
    }

    pub fn release(self: *const Self) void {
        if (self.alloc_ref_count) |alloc_rc| {
            std.debug.print("begin release: {}\n", .{alloc_rc.refCount()});
            if (alloc_rc.count.fetchSub(1, .seq_cst) == 1) {
                std.debug.print("begin real release\n", .{});
                alloc_rc.allocator.free(self.ptr[0..self.len]);
                alloc_rc.allocator.destroy(alloc_rc);
            }
        }
    }

    pub fn sharedView(self: *const Self) Self {
        self.retain();

        return Self{
            .dtype = self.dtype,
            .ptr = self.ptr,
            .len = self.len,
            .alloc_ref_count = self.alloc_ref_count,
        };
    }

    pub fn deepCopy(self: *Self, new_alloc: std.mem.Allocator) !Self {
        const new_ptr = try new_alloc.alloc(u8, self.len);
        @memcpy(new_ptr, self.ptr[0..self.len]);

        return try Self.fromOwned(new_alloc, new_ptr.ptr, new_ptr.len, self.dtype);
    }

    pub fn full(allocator: std.mem.Allocator, element_count: usize, value: anytype) !Self {
        const dtype_i = comptime DType.fromAnyType(@TypeOf(value));
        // const T = comptime dtype_i.toType();

        const buf = try allocator.alloc(u8, element_count);
        // for (buf) |*elem| elem.* = value;

        const dst_buf: []u8 = @ptrCast(buf);

        return try Self.fromOwned(allocator, dst_buf.ptr, dst_buf.len, dtype_i);
    }

    pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Storage {{", .{});
        try writer.print("  count: {},", .{self.dataCount()});
        try writer.print("  ref_count: {d}  ", .{if (self.alloc_ref_count) |rc| rc.refCount() else 0});
        try writer.print("}}", .{});
    }
};

test "storage_basic" {
    const allocator = std.testing.allocator;

    var buf = try RawBuffer.full(allocator, 5, @as(f32, 10.2));
    defer buf.deinit();

    std.debug.print("buf: {f}\n", .{buf});
    // buf.retain();
}

test "atomic" {
    const BufRef = struct {
        ref_count: std.atomic.Value(usize),

        const Self = @This();
        fn retain(self: *Self) usize {
            return self.ref_count.fetchAdd(1, .seq_cst);
        }

        fn release(self: *Self) void {
            if (self.ref_count.fetchSub(1, .seq_cst) == 1) {}
        }
    };

    var buf1 = BufRef{ .ref_count = std.atomic.Value(usize).init(1) };
    std.debug.print("ref count: {}\n", .{buf1.ref_count});
    _ = buf1.retain();

    std.debug.print("ref count: {}\n", .{buf1.ref_count});

    var buf2 = buf1;
    std.debug.print("ref count 3: {}\n", .{buf2.ref_count});
    _ = buf2.retain();
    std.debug.print("ref count 2: {} 4: {}\n", .{ buf1.ref_count, buf2.ref_count });
}
