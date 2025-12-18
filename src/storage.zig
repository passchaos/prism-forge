const std = @import("std");

const RefCount = struct {
    count: usize,
};

pub const Device = enum {
    Cpu,
    Gpu,
};

pub fn Storage(comptime T: type, comptime D: Device) type {
    return struct {
        allocator: std.mem.Allocator,
        _buf: []T,
        _ref_count: *RefCount,

        const Self = @This();

        // generate method
        pub fn arange(allocator: std.mem.Allocator, args: struct {
            start: T = @as(T, 0),
            end: T,
            step: T = @as(T, 1),
        }) !Self {
            var arr_list = try std.ArrayList(T).initCapacity(allocator, 10);

            const start = args.start;
            const step = args.step;

            var tmp = start;
            while (tmp < args.end) {
                try arr_list.append(allocator, tmp);
                tmp += step;
            }

            return try Self.initImpl(allocator, arr_list.items);
        }

        pub fn linspace(allocator: std.mem.Allocator, args: struct {
            start: T,
            end: T,
            steps: usize,
        }) !Self {
            var arr_list = try std.ArrayList(T).initCapacity(allocator, num);

            const start = args.start;
            const end = args.end;
            const steps = args.steps;

            const step = (end - start) / @as(T, num);

            var tmp = start;
            for (0..steps) |i| {
                try arr_list.append(allocator, tmp);
                tmp += step;
            }

            return try Self.initImpl(allocator, arr_list.items);
        }

        // init
        fn initImpl(allocator: std.mem.Allocator, buf: []T) !Self {
            const ref_count = try allocator.create(RefCount);
            ref_count.count = 1;

            return Self{
                .allocator = allocator,
                ._buf = buf,
                ._ref_count = ref_count,
            };
        }

        pub fn dataSlice(self: *const Self) []T {
            return self._buf;
        }

        pub fn bufLen(self: *const Self) usize {
            return self._buf.len;
        }

        pub fn shared(self: *const Self) Self {
            self.retain();

            return Self{
                .allocator = self.allocator,
                ._buf = self._buf,
                ._ref_count = self._ref_count,
            };
        }

        pub fn clone(self: *const Self) Self {
            self.retain();

            return Self{
                .allocator = self.allocator,
                ._buf = self._buf,
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
                ._buf = new_buf.ptr,
                ._ref_count = ref_count,
            };
        }

        pub fn deinit(self: *Self) void {
            self.release();

            if (self._ref_count.count == 0) {
                if (comptime D == .Cpu) {
                    std.debug.print("release storage: {*}\n", .{self._buf.ptr});
                    self.allocator.free(self._buf);
                    self.allocator.destroy(self._ref_count);
                }
            }
        }

        fn retain(self: *const Self) void {
            self._ref_count.count += 1;
        }

        fn release(self: *Self) void {
            if (self._ref_count.count > 0) {
                self._ref_count.count -= 1;
            }
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print("Storage {{\n", .{});
            try writer.print("  device: {},\n", .{D});
            try writer.print("  buf_len: {},\n", .{self.bufLen()});
            try writer.print("  ref_count: {d}\n", .{self._ref_count.count});
            try writer.print("}}\n", .{});
        }
    };
}

test "ref_count" {
    const allocator = std.testing.allocator;

    const StorageF32 = Storage(f32, .Cpu);

    const buf = try allocator.alloc(f32, 10);
    std.debug.print("buf: {*}\n", .{buf.ptr});
    // allocator.free(buf);
    var storage = try StorageF32.initImpl(allocator, buf);
    std.debug.print("storage: {f}\n", .{storage});
    defer storage.deinit();

    try std.testing.expect(storage._ref_count.count == 1);

    storage.retain();
    try std.testing.expect(storage._ref_count.count == 2);

    storage.release();
    try std.testing.expect(storage._ref_count.count == 1);

    storage.release();
    try std.testing.expect(storage._ref_count.count == 0);
}

test "arange" {
    const allocator = std.testing.allocator;

    const StorageI32 = Storage(i32, .Cpu);

    var s1 = try StorageI32.arange(allocator, .{ .end = 10 });
    defer s1.deinit();

    try std.testing.expect(s1.bufLen() == 10);
    try std.testing.expect(s1.dataSlice()[0] == 0);
    try std.testing.expect(s1.dataSlice()[9] == 9);
    std.debug.print("s1: {f}\n", .{s1});
}
