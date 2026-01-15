const std = @import("std");
const utils = @import("utils.zig");

pub fn typeUniqueId(comptime T: type) u64 {
    return std.hash.Wyhash.hash(0, @typeName(T));
}

pub const SymId = u64;

pub const SymbolHandle = struct {
    id: SymId,
    name: []const u8,

    pub fn format(
        self: @This(),
        writer: anytype,
    ) !void {
        try writer.print("Sym{{ id={}, name={s} }}", .{
            self.id,
            self.name,
        });
    }
};

pub fn makeSymbol(comptime sym: anytype) SymbolHandle {
    const T = @TypeOf(sym);
    const name = switch (@typeInfo(T)) {
        .@"struct" => |si| blk: {
            if (@hasField(T, "name")) {
                break :blk @field(sym, "name");
            } else {
                if (si.fields[0].defaultValue()) |name| {
                    break :blk name;
                } else {
                    @compileError("can't find symbol name");
                }
            }
        },
        else => {
            @compileError("don't support this type: " ++ @typeName(T));
        },
    };

    return SymbolHandle{
        .id = typeUniqueId(T),
        .name = name,
    };
}

const Dyn = struct {};

const DimSpec = union(enum) {
    static: usize,
    dyn: Dyn,
    sym: SymbolHandle,
};

const ShapeEnv = struct {
    const Self = @This();

    sym_map: std.AutoHashMap(SymId, usize),
    dyn_map: std.AutoHashMap(usize, usize),

    fn init(allocator: std.mem.Allocator) Self {
        return .{
            .sym_map = std.AutoHashMap(SymId, usize).init(allocator),
            .dyn_map = std.AutoHashMap(usize, usize).init(allocator),
        };
    }

    fn deinit(self: *Self) void {
        self.sym_map.deinit();
        self.dyn_map.deinit();
    }

    fn bind(self: *Self, shape: []const DimSpec, runtime_shape: []const usize) !void {
        for (shape, 0..) |dim, axis| {
            const r_s_v = runtime_shape[axis];
            switch (dim) {
                .static => |v| if (v != r_s_v) return error.ShapeMismatch,
                .dyn => |_| try self.dyn_map.put(axis, r_s_v),
                .sym => |sym| try self.sym_map.put(sym.id, r_s_v),
            }
        }
    }

    fn lookup(self: *const Self, id: SymId) !usize {
        if (self.sym_map.get(id)) |v| return v;

        return error.UnboundSymbol;
    }
};

const SizeExpr = union(enum) {
    const Self = @This();

    Static: usize,
    Sym: SymId,
    Add: struct { lhs: *const Self, rhs: *const Self },
    Mul: struct { lhs: *const Self, rhs: *const Self },

    fn static(v: usize) Self {
        return Self{ .Static = v };
    }

    fn sym(id: SymId) Self {
        return Self{ .Sym = id };
    }

    fn add(lhs: *const Self, rhs: *const Self) Self {
        return Self{ .Add = .{ .lhs = lhs, .rhs = rhs } };
    }

    fn mul(lhs: *const Self, rhs: *const Self) Self {
        return Self{ .Mul = .{ .lhs = lhs, .rhs = rhs } };
    }

    fn eval(self: *const Self, env: *const ShapeEnv) !usize {
        switch (self.*) {
            .Static => |v| v,
            .Sym => |id| env.lookup(id),
            .Add => |add_s| try add_s.lhs.eval(env) + try add_s.rhs.eval(env),
            .Mul => |mul_s| try mul_s.lhs.eval(env) * try mul_s.rhs.eval(env),
        }
    }
};

inline fn parseSpec(comptime dims: anytype) [utils.stt.getFieldsLenComptime(@TypeOf(dims))]DimSpec {
    const N = comptime utils.stt.getFieldsLenComptime(@TypeOf(dims));

    comptime var parsed_dims = [_]DimSpec{undefined} ** N;

    const info = @typeInfo(@TypeOf(dims)).@"struct";

    comptime {
        if (info.is_tuple) {
            for (info.fields, 0..) |f, i| {
                const raw = @field(dims, f.name);

                const T = @TypeOf(raw);

                switch (@typeInfo(T)) {
                    .@"struct" => |_| {
                        parsed_dims[i] = .{ .sym = makeSymbol(raw) };
                    },
                    .comptime_int => {
                        parsed_dims[i] = DimSpec{ .static = raw };
                    },
                    .pointer => {
                        if (utils.str.isString(T)) {
                            // const sd = SymDim(raw);

                            const sd_h = makeSymbol(.{ .name = raw });

                            parsed_dims[i] = .{ .sym = sd_h };
                        } else {
                            @compileError("unsupported type");
                        }
                    },
                    .type => {
                        parsed_dims[i] = .{ .dyn = Dyn{} };
                    },
                    else => {
                        @compileError("unsupported type");
                    },
                }
            }
        } else {
            @compileError("don't support non-tuple");
        }
    }

    return parsed_dims;
}

test "anonymous_struct" {
    {
        const a1 = "test";
        const a2 = "test";

        try std.testing.expect((@TypeOf(a1) == @TypeOf(a2)));
    }
    {
        const a1 = .{"test"};
        const a2 = .{"test"};

        try std.testing.expect((@TypeOf(a1) == @TypeOf(a2)));
    }

    {
        const a1 = .{ .name = "test" };
        const a2 = .{ .name = "test" };

        try std.testing.expect(!(@TypeOf(a1) == @TypeOf(a2)));
    }
}

test "symbol_eql" {
    const sl1 = .{ .name = "s1" };

    const a1 = makeSymbol(sl1);
    const a2 = makeSymbol(sl1);

    try std.testing.expectEqual(a1.id, a2.id);

    const b1 = makeSymbol(.{ .name = "s1" });
    const b2 = makeSymbol(.{ .name = "s1" });
    try std.testing.expect(!(b1.id == b2.id));
}

test "dim_spec" {
    {
        const dim_spec = parseSpec(.{ 2, 10, Dyn, "ddd", "ddd", .{"ddd"}, .{"ddd"}, .{ .name = "ddd" }, .{ .name = "ddd" } });
        const static_v = comptime dim_spec[0].static;
        std.debug.print("dim_spec: {any} {}\n", .{ dim_spec, static_v });
    }

    {
        const dim_spec = parseSpec(.{ 2, Dyn, "ddd" });
        std.debug.print("dim_spec: {any}\n", .{dim_spec});
    }
}

test "broadcast_comp" {}

test "shape creation" {}
