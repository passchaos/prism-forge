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

    // may use egraph to optimize
    pub fn equal(a: Self, b: Self) bool {
        return switch (a) {
            .Static => |a_v| switch (b) {
                .Static => |b_v| a_v == b_v,
                else => |_| false,
            },
            .Sym => |a_s| switch (b) {
                .Sym => |b_s| a_s.id == b_s.id,
                else => |_| false,
            },
            .Add => |a_v| switch (b) {
                .Add => |b_v| a_v.lhs.*.equal(b_v.lhs.*) and a_v.rhs.*.equal(b_v.rhs.*),
                else => |_| false,
            },
            .Mul => |a_v| switch (b) {
                .Mul => |b_v| a_v.lhs.*.equal(b_v.lhs.*) and a_v.rhs.*.equal(b_v.rhs.*),
                else => |_| false,
            },
            .Max => |a_v| switch (b) {
                .Max => |b_v| a_v.lhs.*.equal(b_v.lhs.*) and a_v.rhs.*.equal(b_v.rhs.*),
                else => |_| false,
            },
            .Min => |a_v| switch (b) {
                .Min => |b_v| a_v.lhs.*.equal(b_v.lhs.*) and a_v.rhs.*.equal(b_v.rhs.*),
                else => |_| false,
            },
        };
    }

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

pub fn canBroadcast(
    comptime lhs: []const DimExpr,
    comptime rhs: []const DimExpr,
) bool {
    const max_len = @max(lhs.len, rhs.len);

    inline for (0..max_len) |i| {
        const ldim = if (i < lhs.len) lhs[lhs.len - 1 - i] else DimExpr.static(1);
        const rdim = if (i < rhs.len) rhs[rhs.len - 1 - i] else DimExpr.static(1);

        if (!canBroadcastDim(ldim, rdim)) return false;
    }
    return true;
}

fn canBroadcastDim(comptime a: DimExpr, comptime b: DimExpr) bool {
    return switch (a) {
        .Static => |a_v| switch (b) {
            .Static => |b_v| (a_v == b_v) or (a_v == 1),
            .Sym => |_| a_v == 1,
            else => false,
        },
        else => a.equal(b),
    };
}

pub fn generateBroadcastStride(
    comptime orig_shape_expr: []const DimExpr,
    orig_stride: [orig_shape_expr.len]usize,
    comptime target_shape_expr: []const DimExpr,
) [target_shape_expr.len]usize {
    const N = orig_shape_expr.len;
    const BN = target_shape_expr.len;
    if (N > BN) @compileError("can't broadcast to smaller dimension");

    var new_stride = [_]usize{0} ** BN;

    inline for (0..BN) |t_idx| {
        if (N >= t_idx + 1) {
            const o_dim = orig_shape_expr[N - 1 - t_idx];
            const t_dim = target_shape_expr[BN - 1 - t_idx];

            // @compileLog(std.fmt.comptimePrint("t_idx: {} o_dim: {} t_dim: {}\n", .{ t_idx, o_dim, t_dim }));
            switch (o_dim) {
                .Static => |o_v| {
                    if (o_v == 1) {
                        new_stride[BN - 1 - t_idx] = 0;
                    } else {
                        if (comptime o_dim.equal(t_dim)) {
                            new_stride[BN - 1 - t_idx] = orig_stride[N - 1 - t_idx];
                        } else {
                            @compileError("can't broadcast " ++ std.fmt.comptimePrint("{} != {}", .{ o_dim, t_dim }));
                        }
                    }
                },
                else => {
                    if (comptime o_dim.equal(t_dim)) {
                        new_stride[BN - 1 - t_idx] = orig_stride[N - 1 - t_idx];
                    } else {
                        @compileError("can't broadcast " ++ std.fmt.comptimePrint("{} != {}", .{ o_dim, t_dim }));
                    }
                },
            }
        } else {
            new_stride[BN - 1 - t_idx] = 0;
        }
    }

    return new_stride;
}

pub fn compatibleBroacastShapes(comptime lhs_shape: []const usize, comptime rhs_shape: []const usize) [@max(lhs_shape.len, rhs_shape.len)]usize {
    const l_l = lhs_shape.len;
    const r_l = rhs_shape.len;

    if (l_l == 0 or r_l == 0) @compileError("can't use zero-d tensor to broadcast");

    const result_len = @max(l_l, r_l);

    comptime var result = [_]usize{0} ** result_len;

    comptime {
        for (0..result_len) |i| {
            const v = if (l_l > i) blk: {
                const v = if (r_l > i) @max(lhs_shape[l_l - i - 1], rhs_shape[r_l - i - 1]) else lhs_shape[l_l - i - 1];
                break :blk v;
            } else rhs_shape[r_l - i - 1];

            result[result_len - i - 1] = v;
        }
    }

    return result;
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

test "broadcast_comp" {
    const lhs = comptime DimExpr.static(1);

    const lhs_1 = comptime DimExpr.static(2);
    const rhs = comptime DimExpr.static(3);

    try std.testing.expect(canBroadcastDim(lhs, rhs));
    try std.testing.expect(!canBroadcastDim(lhs_1, rhs));

    {
        const exprs_1 = comptime parseSpec(.{ 2, 3, 4, .{ .name = "ddd" } });
        const exprs_2 = comptime parseSpec(.{ 2, 3, 4, .{ .name = "ddd" } });
        try std.testing.expect(!canBroadcast(&exprs_1, &exprs_2));
    }

    {
        const exprs_1 = comptime parseSpec(.{ 2, 3, 4, "ddd" });
        const exprs_2 = comptime parseSpec(.{ 2, 3, 4, "ddd" });
        std.debug.print("exprs_1: {any}\n", .{exprs_1});
        std.debug.print("exprs_2: {any}\n", .{exprs_2});
        try std.testing.expect(canBroadcast(&exprs_1, &exprs_2));
    }

    {
        const exprs_1 = comptime parseSpec(.{ 2, 3, 4, "ddd" });
        const stride_1 = [4]usize{ 1, 2, 3, 4 };
        const exprs_2 = comptime parseSpec(.{ 1, 3, 1, "ddd" });

        const stride_2 = generateBroadcastStride(&exprs_2, stride_1, &exprs_1);
        std.debug.print("stride_2: {any}\n", .{stride_2});
    }
}

test "shape creation" {}
