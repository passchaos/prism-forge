const std = @import("std");

pub const Device = enum { Cpu, Cuda };

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
