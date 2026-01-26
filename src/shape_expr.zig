const std = @import("std");
const utils = @import("utils.zig");

pub fn typeUniqueId(comptime T: type) u64 {
    const signature = switch (@typeInfo(T)) {
        .@"struct" => utils.stt.structSignature(T),
        else => @typeName(T),
    };
    return std.hash.Wyhash.hash(0, signature);
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
    lhs: *const SizeExpr,
    rhs: *const SizeExpr,
};

pub fn staticShapeExpr(comptime shape: []const usize) [shape.len]SizeExpr {
    const N = shape.len;

    var result: [N]SizeExpr = undefined;
    for (shape, 0..) |s, i| {
        result[i] = SizeExpr.static(s);
    }
    return result;
}

pub const SizeExpr = union(enum) {
    const Self = @This();

    Static: usize,
    Sym: SymbolHandle,
    Add: BinaryDimOpExpr,
    Mul: BinaryDimOpExpr,
    Max: BinaryDimOpExpr,
    Min: BinaryDimOpExpr,

    pub fn format(self: @This(), writer: *std.Io.Writer) !void {
        switch (self) {
            .Static => |v| try writer.print("{d}", .{v}),
            .Sym => |s| try writer.print("\"{s}\"", .{s.name}),
            .Add => |op| try writer.print("({f} + {f})", .{ op.lhs, op.rhs }),
            .Mul => |op| try writer.print("({f} * {f})", .{ op.lhs, op.rhs }),
            .Max => |op| try writer.print("max({f}, {f})", .{ op.lhs, op.rhs }),
            .Min => |op| try writer.print("min({f}, {f})", .{ op.lhs, op.rhs }),
        }
    }

    pub fn static_value(self: *const Self) usize {
        return switch (self.*) {
            .Static => |v| v,
            else => @compileError("Not static"),
        };
    }

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
            else => false,
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
        return switch (self.*) {
            .Static => |v| v,
            .Sym => |s| env.lookup(&s),
            .Add => |add_s| try add_s.lhs.eval(env) + try add_s.rhs.eval(env),
            .Mul => |mul_s| try mul_s.lhs.eval(env) * try mul_s.rhs.eval(env),
            .Max => |max_s| @max(try max_s.lhs.eval(env), try max_s.rhs.eval(env)),
            .Min => |min_s| @min(try min_s.lhs.eval(env), try min_s.rhs.eval(env)),
        };
    }
};

const SymbolHandleHash = struct {
    pub fn hash(_: @This(), key: *const SymbolHandle) u64 {
        return key.id;
    }

    pub fn eql(_: @This(), a: *const SymbolHandle, b: *const SymbolHandle) bool {
        return a.id == b.id;
    }
};

const SymbolHandleHashMap = std.HashMap(
    *const SymbolHandle,
    usize,
    SymbolHandleHash,
    80,
);

const SymbolHandleHashSet = std.HashMap(
    *const SymbolHandle,
    void,
    SymbolHandleHash,
    80,
);

pub const ShapeEnv = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    sym_map: SymbolHandleHashMap,
    globals: SymbolHandleHashSet,
    scopes: std.ArrayList(SymbolHandleHashSet),

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.print("ShapeEnv {{\n", .{});

        try writer.print("  syms: {{,\n", .{});
        {
            var map_iter = self.sym_map.iterator();
            while (map_iter.next()) |entry| {
                try writer.print("    {s}: {},\n", .{
                    entry.key_ptr.*.name,
                    entry.value_ptr.*,
                });
            }
        }
        try writer.print("  }}\n", .{});
        try writer.print("  globals: {{,\n", .{});
        {
            var global_iter = self.globals.iterator();
            try writer.print("   ", .{});
            while (global_iter.next()) |entry| {
                try writer.print(" \"{s}\"", .{
                    entry.key_ptr.*.name,
                });
            }
        }
        try writer.print("\n  }}\n", .{});

        try writer.print("  scopes: {{\n", .{});
        {
            for (self.scopes.items, 0..) |scope, i| {
                var scope_iter = scope.iterator();

                try writer.print("    {}:", .{i});
                while (scope_iter.next()) |entry| {
                    try writer.print(" \"{s}\"", .{
                        entry.key_ptr.*.name,
                    });
                }
                try writer.print("\n", .{});
            }
        }
        try writer.print("  }}\n", .{});

        try writer.print("}}\n", .{});
    }

    pub fn init(allocator: std.mem.Allocator) !Self {
        return .{
            .allocator = allocator,
            .sym_map = SymbolHandleHashMap.init(allocator),
            .globals = SymbolHandleHashSet.init(allocator),
            .scopes = try std.ArrayList(SymbolHandleHashSet).initCapacity(allocator, 10),
        };
    }

    pub fn reset(self: *Self) void {
        self.sym_map.clearAndFree();
        self.globals.clearAndFree();
        for (self.scopes) |scope| {
            scope.clearAndFree();
        }
        self.scopes.clearAndFree(self.allocator);
    }

    pub fn deinit(self: *Self) void {
        self.reset();
        self.sym_map.deinit();
    }

    pub fn pushScope(self: *Self) !void {
        try self.scopes.append(self.allocator, SymbolHandleHashSet.init(self.allocator));
    }

    pub fn popScope(self: *Self) void {
        if (self.scopes.items.len == 0) @panic("no scope found");
        if (self.scopes.pop()) |scope| {
            var iter = scope.keyIterator();
            while (iter.next()) |k| {
                _ = self.sym_map.remove(k.*);
            }

            {
                var ss = scope;
                ss.clearAndFree();
            }
        }
    }

    fn bindInner(self: *Self, sym: *const SymbolHandle, value: usize) !void {
        if (self.sym_map.get(sym)) |v| {
            if (v != value) return error.CannotRebindSymbolOtherValue;
        } else {
            try self.sym_map.put(sym, value);
        }
    }

    pub fn bindGlobal(self: *Self, sym: *const SymbolHandle, value: usize) !void {
        // std.debug.print("bind global: sym= {f} value= {}\n", .{ sym, value });

        try self.bindInner(sym, value);
        try self.globals.put(sym, void{});
    }

    pub fn bind(self: *Self, sym: *const SymbolHandle, value: usize) !void {
        // std.debug.print("bind scope: sym= {f} value= {}\n", .{ sym, value });

        try self.bindInner(sym, value);

        const scope_len = self.scopes.items.len;
        if (scope_len == 0) {
            try self.pushScope();
        }

        if (scope_len > 0) {
            try self.scopes.items[scope_len - 1].put(sym, void{});
        }
    }

    pub fn lookup(self: *const Self, sym: *const SymbolHandle) !usize {
        if (self.sym_map.get(sym)) |v| return v;

        return error.UnboundSymbol;
    }

    pub fn lookupShape(self: *const Self, comptime shape_expr: []const SizeExpr) ![shape_expr.len]usize {
        var result: [shape_expr.len]usize = undefined;
        for (shape_expr, 0..) |dim_expr, i| {
            result[i] = try dim_expr.eval(self);
        }
        return result;
    }
};

pub fn product(dim_exprs: []const SizeExpr) SizeExpr {
    var result = SizeExpr.static(1);
    for (dim_exprs) |dim| {
        result = switch (result) {
            .Static => |rs| switch (dim) {
                .Static => |ds| SizeExpr.static(rs * ds),
                else => SizeExpr.mul(&result, &dim),
            },
            else => SizeExpr.mul(&result, &dim),
        };
    }
    return result;
}

pub fn insertDimComptime(
    comptime arr: []const SizeExpr,
    comptime dim: usize,
    comptime value: usize,
) [arr.len + 1]SizeExpr {
    const N = arr.len;

    if (dim > N) @compileError("Invalid dimension");

    if (N == 0) {
        return [_]usize{SizeExpr.static(value)};
    } else {
        var new_arr = [_]SizeExpr{undefined} ** (N + 1);

        var i: usize = 0;
        var j: usize = 0;

        while (i < N + 1) {
            if (i == dim) {
                new_arr[i] = SizeExpr.static(value);
                i += 1;
                continue;
            }
            new_arr[i] = arr[j];
            i += 1;
            j += 1;
        }

        return new_arr;
    }
}

pub fn removeDimComptime(comptime arr: []const SizeExpr, comptime dim: usize) [arr.len - 1]SizeExpr {
    const N = arr.len;

    if (dim >= N) @compileError("Invalid dimension");
    if (N == 0) @compileError("don't support 0-1-d tensor removeDim op");
    if (N == 1) return [0]usize{};

    if (N == 0) {
        @compileError("don't support 0-1-d tensor removeDim op");
    } else if (N == 1) {
        return [0]usize{};
    } else {
        switch (arr[dim]) {
            .Static => |sv| if (sv != 1) @compileError(
                "Dim not one" ++ std.fmt.comptimePrint("{}", .{sv}),
            ),
            else => @compileError("Dim not static one"),
        }

        var new_arr = [_]SizeExpr{undefined} ** (N - 1);
        var i: usize = 0;
        var j: usize = 0;

        while (i < N) {
            if (i == dim) {
                i += 1;
                continue;
            }
            new_arr[j] = arr[i];
            i += 1;
            j += 1;
        }

        return new_arr;
    }
}

pub fn parseSpec(comptime dims: anytype) [utils.stt.getFieldsLenComptime(@TypeOf(dims))]SizeExpr {
    const N = comptime utils.stt.getFieldsLenComptime(@TypeOf(dims));

    comptime var parsed_dims = [_]SizeExpr{undefined} ** N;

    const info = @typeInfo(@TypeOf(dims)).@"struct";

    comptime {
        if (info.is_tuple) {
            for (info.fields, 0..) |f, i| {
                const raw = @field(dims, f.name);

                const T = @TypeOf(raw);

                if (T == SizeExpr) {
                    parsed_dims[i] = raw;
                    continue;
                }

                switch (@typeInfo(T)) {
                    .@"struct" => |_| {
                        parsed_dims[i] = SizeExpr.sym(raw);
                    },
                    .comptime_int => {
                        parsed_dims[i] = SizeExpr.static(raw);
                    },
                    .pointer => {
                        if (utils.str.isString(T)) {
                            parsed_dims[i] = SizeExpr.sym(.{ .name = raw });
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

pub fn allStatic(comptime shape_expr: []const SizeExpr) bool {
    inline for (shape_expr) |dim| {
        switch (dim) {
            .Static => continue,
            else => return false,
        }
    }
    return true;
}

pub fn shapeExprEqual(
    comptime lhs: []const SizeExpr,
    comptime rhs: []const SizeExpr,
) bool {
    if (lhs.len != rhs.len) return false;
    inline for (lhs, rhs) |l, r| {
        if (!l.equal(r)) return false;
    }
    return true;
}

pub fn canBroadcast(
    comptime lhs: []const SizeExpr,
    comptime rhs: []const SizeExpr,
) bool {
    const max_len = @max(lhs.len, rhs.len);

    inline for (0..max_len) |i| {
        const ldim = if (i < lhs.len) lhs[lhs.len - 1 - i] else SizeExpr.static(1);
        const rdim = if (i < rhs.len) rhs[rhs.len - 1 - i] else SizeExpr.static(1);

        if (!canBroadcastDim(ldim, rdim)) return false;
    }
    return true;
}

fn canBroadcastDim(comptime a: SizeExpr, comptime b: SizeExpr) bool {
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
    comptime orig_shape_expr: []const SizeExpr,
    orig_stride: [orig_shape_expr.len]usize,
    comptime target_shape_expr: []const SizeExpr,
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
    {
        const ss1 = comptime SizeExpr.static(10);
        const ss_v = comptime ss1.static_value();
        try std.testing.expect(ss_v == 10);
        // @compileLog(std.fmt.comptimePrint("ss_v: {}\n", .{ss_v}));
    }

    const sym1 = comptime makeSymbol(.{ .name = "abc" });

    const sym12 = comptime SizeExpr.add(&SizeExpr{ .Sym = sym1 }, &SizeExpr.sym(.{ .name = "abc" }));

    {
        const dim_spec = comptime parseSpec(.{ sym12, 2, 10, "ddd", "ddd", .{"ddd"}, .{"ddd"}, .{ .name = "ddd" }, .{ .name = "ddd" } });
        const static_v = comptime dim_spec[1].Static;
        std.debug.print("dim_spec: {any} {}\n", .{ dim_spec, static_v });
    }

    {
        const dim_spec = parseSpec(.{ 2, "ddd" });
        std.debug.print("dim_spec: {any}\n", .{dim_spec});
    }
}

test "broadcast_comp" {
    const lhs = comptime SizeExpr.static(1);

    const lhs_1 = comptime SizeExpr.static(2);
    const rhs = comptime SizeExpr.static(3);

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
