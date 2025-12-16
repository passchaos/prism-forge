const std = @import("std");

pub const Device = enum { Cpu, Cuda };

const RefCount = struct {
    count: usize,
};

allocator: std.mem.Allocator,
_device: Device,
_buf: [*]u8,
_bytes_size: usize,
_ref_count: *RefCount,

const Self = @This();

pub fn init(allocator: std.mem.Allocator, device: Device, buf: [*]u8, bytes_size: usize) Self {
    const ref_count = allocator.create(RefCount) catch unreachable;
    ref_count.count = 1;

    return Self{
        .allocator = allocator,
        ._device = device,
        ._buf = buf,
        ._bytes_size = bytes_size,
        ._ref_count = ref_count,
    };
}

pub fn dataSlice(self: *const Self, comptime T: anytype) []T {
    const d_buf: [*]T = @ptrCast(@alignCast(self._buf));

    return d_buf[0 .. self.byteSize() / @sizeOf(T)];
}

pub fn rawDataSlice(self: *const Self) [*]u8 {
    return self._buf;
}

pub fn byteSize(self: *const Self) usize {
    return self._bytes_size;
}

pub fn clone(self: *const Self) Self {
    self.retain();

    return Self{
        .allocator = self.allocator,
        ._device = self._device,
        ._buf = self._buf,
        ._bytes_size = self._bytes_size,
        ._ref_count = self._ref_count,
    };
}

pub fn deepCopy(self: *const Self) !Self {
    const new_buf = try self.allocator.alloc(u8, self._bytes_size);
    @memcpy(new_buf, self._buf);

    const ref_count = try self.allocator.create(RefCount);
    ref_count.count = 1;

    return Self{
        .allocator = self.allocator,
        ._device = self._device,
        ._buf = new_buf.ptr,
        ._bytes_size = self._bytes_size,
        ._ref_count = ref_count,
    };
}

pub fn deinit(self: *Self) void {
    self.release();
}

fn retain(self: *const Self) void {
    self._ref_count.count += 1;
}

fn release(self: *Self) void {
    if (self._ref_count.count > 0) {
        self._ref_count.count -= 1;
    }

    if (self._ref_count == 0) {
        if (self._device == .Cpu) {
            if (self._buf) |buf| {
                std.debug.print("release storage\n", .{});
                self.allocator.free(buf);
            }
        }
    }
}

pub fn format(
    self: @This(),
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.print("Storage {{\n", .{});
    try writer.print("  device: {any},\n", .{self._device});
    try writer.print("  bytes_size: {d},\n", .{self._bytes_size});
    try writer.print("  ref_count: {d}\n", .{self._ref_count.count});
    try writer.print("}}\n", .{});
}
