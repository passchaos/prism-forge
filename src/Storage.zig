const std = @import("std");

pub const Device = enum { Cpu, Cuda };

allocator: std.mem.Allocator,
_device: Device,
_buf: [*]u8,
_bytes_size: usize,
_ref_count: usize,

const Self = @This();

pub fn init(allocator: std.mem.Allocator, device: Device, buf: [*]u8, bytes_size: usize) Self {
    return Self{
        .allocator = allocator,
        ._device = device,
        ._buf = buf,
        ._bytes_size = bytes_size,
        ._ref_count = 1,
    };
}

pub fn dataSlice(self: *const Self, comptime T: anytype) [*]T {
    const d_buf: [*]T = @ptrCast(@alignCast(self._buf));

    return d_buf;
}

pub fn byteSize(self: *const Self) usize {
    return self._bytes_size;
}

pub fn clone(self: *const Self) Self {
    const v_s = @constCast(self);
    v_s.retain();

    return Self{
        .allocator = self.allocator,
        ._device = self._device,
        ._buf = self._buf,
        ._bytes_size = self._bytes_size,
        ._ref_count = self._ref_count,
    };
}

pub fn deinit(self: *Self) void {
    self.release();
}

fn retain(self: *Self) void {
    self._ref_count += 1;
}

fn release(self: *Self) void {
    if (self._ref_count > 0) {
        self._ref_count -= 1;
    }

    if (self._ref_count == 0) {
        if (self._device == .Cpu) {
            if (self._buf) |buf| {
                self.allocator.free(buf);
            }
        }
    }
}
