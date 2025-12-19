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

        pub fn cat(allocator: std.mem.Allocator, storages: []const Self) !Self {
            var total_len: usize = 0;
            for (storages) |storage| {
                total_len += storage.len();
            }

            var buf = try allocator.alloc(T, total_len);
            var offset: usize = 0;
            for (storages) |storage| {
                @memcpy(buf[offset .. offset + storage.len()], storage._buf);
                offset += storage.len();
            }

            return try Self.initImpl(allocator, buf);
        }

        // generate method
        pub fn arange(allocator: std.mem.Allocator, args: struct {
            start: T = @as(T, 0),
            step: T = @as(T, 1),
            end: T,
        }) !Self {
            var arr_list = try std.ArrayList(T).initCapacity(allocator, 10);

            const start = args.start;
            const step = args.step;

            var tmp = start;
            while (tmp < args.end) {
                try arr_list.append(allocator, tmp);
                tmp += step;
            }

            return try Self.initImpl(allocator, try arr_list.toOwnedSlice(allocator));
        }

        pub fn linspace(allocator: std.mem.Allocator, args: struct {
            start: T,
            end: T,
            steps: usize,
        }) !Self {
            switch (@typeInfo(T)) {
                .float => {
                    const start = args.start;
                    const end = args.end;
                    const num = args.steps;

                    var buf = try allocator.alloc(T, num);

                    const step = (end - start) / @as(T, @floatFromInt(num));

                    var tmp = start;
                    for (0..num) |i| {
                        buf[i] = tmp;
                        tmp += step;
                    }

                    return try Self.initImpl(allocator, buf);
                },
                else => @compileError("Unsupported data type " ++ @typeName(T)),
            }
        }

        pub fn full(allocator: std.mem.Allocator, element_count: usize, value: T) !Self {
            const buf = try allocator.alloc(T, element_count);
            for (buf) |*elem| elem.* = value;
            return try Self.initImpl(allocator, buf);
        }

        pub fn rand(allocator: std.mem.Allocator, element_count: usize, low: T, high: T) !Self {
            if (comptime T != f32 and T != f64) {
                @compileError("Unsupported type" ++ @typeName(T));
            }

            const buf = try allocator.alloc(T, element_count);

            var rpng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
            const rng = rpng.random();

            for (buf) |*elem| {
                const u = rng.float(T);
                elem.* = low + (high - low) * u;
            }

            return try Self.initImpl(allocator, buf);
        }

        pub fn randNorm(allocator: std.mem.Allocator, element_count: usize, mean_a: T, stddev: T) !Self {
            if (comptime T != f32 and T != f64) {
                @compileError("Unsupported type" ++ @typeName(T));
            }

            const buf = try allocator.alloc(T, element_count);

            var rpng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
            const rng = rpng.random();

            for (buf) |*elem| {
                const u = rng.floatNorm(T);
                elem.* = mean_a + stddev * u;
            }

            return try Self.initImpl(allocator, buf);
        }

        // init
        pub fn initImpl(allocator: std.mem.Allocator, buf: []T) !Self {
            const ref_count = try allocator.create(RefCount);
            ref_count.count = 1;
            std.debug.print("init storage: buf= {*}\n", .{buf.ptr});

            return Self{
                .allocator = allocator,
                ._buf = buf,
                ._ref_count = ref_count,
            };
        }

        pub fn dataSlice(self: *const Self) []T {
            return self._buf;
        }

        pub fn len(self: *const Self) usize {
            return self._buf.len;
        }

        pub fn refCount(self: *const Self) usize {
            return self._ref_count.count;
        }

        pub fn shared(self: *const Self) Self {
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

        pub fn deinit(self: *const Self) void {
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

        fn release(self: *const Self) void {
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
            try writer.print("  buf_len: {},\n", .{self.len()});
            try writer.print("  ref_count: {d}\n", .{self._ref_count.count});
            try writer.print("}}", .{});
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
    defer storage.deinit();

    {
        const s1 = storage.shared();
        defer s1.deinit();
        try std.testing.expectEqual(s1.refCount(), 2);

        const s2 = storage.shared();
        defer s2.deinit();
        try std.testing.expectEqual(s1.refCount(), 3);

        const s3 = storage.shared();
        defer s3.deinit();
        try std.testing.expectEqual(s1.refCount(), 4);

        const s4 = storage.shared();
        defer s4.deinit();
        try std.testing.expectEqual(s1.refCount(), 5);

        std.debug.print("storage: {f}\ns1: {f}\ns2: {f}\ns3: {f}\ns4: {f}\n", .{ storage, s1, s2, s3, s4 });
    }

    try std.testing.expect(storage.refCount() == 1);

    storage.retain();
    try std.testing.expect(storage.refCount() == 2);

    storage.release();
    try std.testing.expect(storage.refCount() == 1);

    storage.release();
    try std.testing.expect(storage.refCount() == 0);
}

test "arange" {
    const allocator = std.testing.allocator;

    const StorageI32 = Storage(i32, .Cpu);

    var s1 = try StorageI32.arange(allocator, .{ .end = 10 });
    defer s1.deinit();

    try std.testing.expect(s1.len() == 10);
    try std.testing.expect(s1.dataSlice()[0] == 0);
    try std.testing.expect(s1.dataSlice()[9] == 9);
    std.debug.print("s1: {f}\n", .{s1});
}

test "linspace" {
    const allocator = std.testing.allocator;

    const StorageI32 = Storage(f32, .Cpu);

    var s1 = try StorageI32.linspace(allocator, .{
        .start = 1,
        .steps = 10,
        .end = 30,
    });
    defer s1.deinit();

    std.debug.print("s1: {f} data_slice: {any}\n", .{ s1, s1.dataSlice() });

    try std.testing.expect(s1.len() == 10);
}

test "random" {
    const allocator = std.testing.allocator;
    const StorageF32 = Storage(f32, .Cpu);

    const s1 = try StorageF32.rand(allocator, 100, -3.0, 7.0);
    defer s1.deinit();
    const s2 = try StorageF32.randNorm(allocator, 100, -3.0, 7.0);
    defer s2.deinit();

    try std.testing.expect(s1.len() == 100);
    try std.testing.expect(s2.len() == 100);

    std.debug.print("s1: {f} s2: {f}\n", .{ s1, s2 });
}

test "cat" {
    const allocator = std.testing.allocator;

    const StorageF32 = Storage(f32, .Cpu);

    const s1 = try StorageF32.rand(allocator, 10, -3, 3);
    defer s1.deinit();
    const s2 = try StorageF32.randNorm(allocator, 20, 3.0, 1.0);
    defer s2.deinit();

    const sc = try StorageF32.cat(allocator, &.{ s1, s2 });
    defer sc.deinit();

    try std.testing.expectEqualSlices(f32, sc.dataSlice()[0..s1.len()], s1.dataSlice());
    try std.testing.expectEqualSlices(f32, sc.dataSlice()[s1.len()..], s2.dataSlice());
    std.debug.print("s1: {any}\ns2: {any}\nsc: {any}\n", .{ s1.dataSlice(), s2.dataSlice(), sc.dataSlice() });
}
