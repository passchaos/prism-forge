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

const BinaryDimOpExpr = struct {
    lhs: *const DimExpr,
    rhs: *const DimExpr,
};

pub const DimExpr = union(enum) {
    const Self = @This();

    Static: usize,
    Sym: SymbolHandle,
    Add: BinaryDimOpExpr,
    Mul: BinaryDimOpExpr,
    Max: BinaryDimOpExpr,
    Min: BinaryDimOpExpr,

    pub fn static(v: usize) Self {
        return Self{ .Static = v };
    }

    pub fn sym(comptime sym_v: anytype) Self {
        return Self{ .Sym = makeSymbol(sym_v) };
    }

    pub fn add(lhs: *const Self, rhs: *const Self) Self {
        return Self{ .Add = .{ .lhs = lhs, .rhs = rhs } };
    }

    pub fn mul(lhs: *const Self, rhs: *const Self) Self {
        return Self{ .Mul = .{ .lhs = lhs, .rhs = rhs } };
    }

    pub fn max(lhs: *const Self, rhs: *const Self) Self {
        return Self{ .Max = .{ .lhs = lhs, .rhs = rhs } };
    }

    pub fn min(lhs: *const Self, rhs: *const Self) Self {
        return Self{ .Min = .{ .lhs = lhs, .rhs = rhs } };
    }

    pub fn eval(self: *const Self, env: *const ShapeEnv) !usize {
        switch (self.*) {
            .Static => |v| v,
            .Sym => |id| env.lookup(id),
            .Add => |add_s| try add_s.lhs.eval(env) + try add_s.rhs.eval(env),
            .Mul => |mul_s| try mul_s.lhs.eval(env) * try mul_s.rhs.eval(env),
            .Max => |max_s| @max(try max_s.lhs.eval(env), try max_s.rhs.eval(env)),
            .Min => |min_s| @min(try min_s.lhs.eval(env), try min_s.rhs.eval(env)),
        }
    }
};

pub const ShapeEnv = struct {
    const Self = @This();

    sym_map: std.AutoHashMap(SymId, usize),

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .sym_map = std.AutoHashMap(SymId, usize).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.sym_map.deinit();
    }

    pub fn bind(self: *Self, id: SymId, value: usize) !void {
        try self.sym_map.put(id, value);
    }

    pub fn lookup(self: *const Self, id: SymId) !usize {
        if (self.sym_map.get(id)) |v| return v;

        return error.UnboundSymbol;
    }
};

pub fn product(dim_exprs: []const DimExpr) DimExpr {
    var result = DimExpr.static(1);
    for (dim_exprs) |dim| {
        result = DimExpr.mul(result, dim);
    }
    return result;
}

pub inline fn parseSpec(comptime dims: anytype) [utils.stt.getFieldsLenComptime(@TypeOf(dims))]DimExpr {
    const N = comptime utils.stt.getFieldsLenComptime(@TypeOf(dims));

    comptime var parsed_dims = [_]DimExpr{undefined} ** N;

    const info = @typeInfo(@TypeOf(dims)).@"struct";

    comptime {
        if (info.is_tuple) {
            for (info.fields, 0..) |f, i| {
                const raw = @field(dims, f.name);

                const T = @TypeOf(raw);

                if (T == DimExpr) {
                    parsed_dims[i] = raw;
                    continue;
                }

                switch (@typeInfo(T)) {
                    .@"struct" => |_| {
                        parsed_dims[i] = DimExpr.sym(raw);
                    },
                    .comptime_int => {
                        parsed_dims[i] = DimExpr.static(raw);
                    },
                    .pointer => {
                        if (utils.str.isString(T)) {
                            parsed_dims[i] = DimExpr.sym(.{ .name = raw });
                        } else {
                            @compileError("unsupported type");
                        }
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
    const sym1 = comptime makeSymbol(.{ .name = "abc" });

    const sym12 = comptime DimExpr.add(&DimExpr{ .Sym = sym1 }, &DimExpr.sym(.{ .name = "abc" }));

    {
        const dim_spec = parseSpec(.{ sym12, 2, 10, "ddd", "ddd", .{"ddd"}, .{"ddd"}, .{ .name = "ddd" }, .{ .name = "ddd" } });
        const static_v = comptime dim_spec[1].Static;
        std.debug.print("dim_spec: {any} {}\n", .{ dim_spec, static_v });
    }

    {
        const dim_spec = parseSpec(.{ 2, "ddd" });
        std.debug.print("dim_spec: {any}\n", .{dim_spec});
    }
}

test "broadcast_comp" {}

test "shape creation" {}
