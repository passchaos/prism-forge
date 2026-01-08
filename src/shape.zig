const std = @import("std");
const utils = @import("utils.zig");

// const TypeDim = union(enum) {
//     static: usize,
//     named: []const u8,
// };

/// 通用ID生成器：同时支持编译时（comptime）和运行时（runtime）
pub const IDGenerator = struct {
    // 编译时计数器（仅编译时可见，无原子操作）
    comptime_counter: u64 = 0,
    // 运行时原子计数器（仅运行时生效）
    runtime_counter: std.atomic.Atomic(u64) = std.atomic.Atomic(u64).init(0),

    /// 通用next方法：自动适配编译时/运行时
    pub fn next(self: *@This()) u64 {
        if (@inComptime()) {
            // 编译时：直接自增编译时计数器（无并发，无需原子操作）
            const id = self.comptime_counter;
            self.comptime_counter += 1;
            return id;
        } else {
            // 运行时：使用原子操作自增（线程安全）
            const id = self.runtime_counter.fetchAdd(1, .SeqCst);
            return id;
        }
    }

    /// 重置计数器（编译时/运行时均生效）
    pub fn reset(self: *@This()) void {
        if (@inComptime()) {
            self.comptime_counter = 0;
        } else {
            self.runtime_counter.store(0, .SeqCst);
        }
    }
};

pub const DictEntry = struct {
    name: []const u8,
    value: usize,
};

pub fn isString(comptime st: type) bool {
    const ti = @typeInfo(st);

    switch (ti) {
        .pointer => |pi| if (pi.is_const and pi.size == .one) {
            switch (@typeInfo(pi.child)) {
                .array => |ai| {
                    if (ai.sentinel()) |stl| {
                        if (stl == 0 and ai.child == u8) {
                            return true;
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                },
                else => return false,
            }
        },
        else => return false,
    }
}

fn getSpecDimSymbol(comptime dim: anytype) []const u8 {
    switch (@typeInfo(@TypeOf(dim))) {
        .pointer => |pi| {
            if (!pi.is_const or pi.size != .one)
                @compileError("shape: pointer must be string literal " ++ @typeName(@TypeOf(dim)) ++ " " ++ @typeName(pi.child));

            switch (@typeInfo(pi.child)) {
                .array => |ai| {
                    if (ai.sentinel()) |_| {
                        return dim;
                    } else {
                        @compileError("shape: pointer must be string literal");
                    }
                },
                else => @compileError("shape: unsupported pointer type"),
            }
        },
        else => @compileError("shape: unsupported type"),
    }
}

fn evalRuntimeDim(comptime dim: anytype, comptime dict: []const DictEntry) usize {
    if (@TypeOf(dim) == usize or @TypeOf(dim) == comptime_int) return dim;

    // const name = getSymbol(dim);
    comptime {
        for (dict) |entry| {
            // @compileLog(std.fmt.comptimePrint("entry: {s} dim: {s}", .{ entry.name, dim }));
            if (std.mem.eql(u8, entry.name, dim))
                // return 8;
                return entry.value;
        }
        @compileError("Missing runtime value for named dimension: " ++ dim);
    }

    // return getSymbol(dim);

    //     if (@TypeOf(...))

    //     return switch (dim) {
    //         usize => |v| v,

    // .pointer => |p| blk: {
    //                 if (!p.is_const or p.size != .one)
    //                     @compileError("shape: pointer must be string literal " ++ @typeName(T) ++ " " ++ @typeName(p.child));

    //                 switch (@typeInfo(p.child)) {
    //                     .array => |ai| {
    //                         if (ai.sentinel()) |_| {
    //                             break :blk .{ .named = raw[0..] };
    //                         } else {
    //                             @compileError("shape: pointer must be string literal");
    //                         }
    //                     },
    //                     else => @compileError("shape: unsupported pointer type"),
    //                 }
    //             },         => |name| blk: {
    //             comptime {
    //                 for (dict) |entry| {
    //                     if (std.mem.eql(u8, entry.name, name))
    //                         break :blk entry.value; // runtime OK
    //                 }
    //                 @compileError("Missing runtime value for named dimension: " ++ name);
    //             }
    //         },
    //     };
}

pub fn evalShape(comptime spec: type, comptime dict: []const DictEntry) [@typeInfo(spec).@"struct".fields.len]usize {
    const info = @typeInfo(spec).@"struct";
    const N = info.fields.len;

    var out: [N]usize = undefined;

    inline for (info.fields, 0..) |f, i| {
        // const ptr_any = f.default_value_ptr orelse @compileError("Missing TypeDim");
        // const td_ptr = @as(*const TypeDim, @ptrCast(@alignCast(ptr_any)));
        // const td = td_ptr.*;

        // @compileLog("f name: " ++ f.name);
        const td = f.defaultValue() orelse @compileError("Missing default value");

        out[i] = comptime evalRuntimeDim(td, dict);
    }

    return out;
}

pub fn dimsLen(comptime spec: type) usize {
    return @typeInfo(spec).@"struct".fields.len;
}

pub fn allStatic(comptime spec: type) bool {
    return staticDimLen(spec) == dimsLen(spec);
}

pub fn staticDimLen(comptime spec: type) usize {
    const info = @typeInfo(spec).@"struct";

    var count: usize = 0;
    inline for (info.fields) |f| {
        if (f.type == usize or f.type == comptime_int) count += 1;
    }

    return count;
}

pub fn staticDims(comptime spec: type) [staticDimLen(spec)]usize {
    const info = @typeInfo(spec).@"struct";

    var out: [staticDimLen(spec)]usize = undefined;

    var d: usize = 0;
    inline for (info.fields, 0..) |f, i| {
        if (f.type == usize or f.type == comptime_int) {
            out[d] = i;
            d += 1;
        }
    }

    return out;
}

pub fn generateBroadcastStride(
    comptime l_spec: anytype,
    comptime r_spec: anytype,
    l_shape: [staticDimLen(l_spec)]usize,
    l_stride: [staticDimLen(l_spec)]usize,
    r_shape: [staticDimLen(r_spec)]usize,
    r_stride: [staticDimLen(r_spec)]usize,
) [staticDimLen(r_spec)]usize {
    const info = @typeInfo(@TypeOf(lts)).@"struct";

    var out: [staticDimLen(lts)]usize = undefined;

    var d: usize = 0;
    inline for (info.fields, 0..) |f, i| {
        if (f.type == usize or f.type == comptime_int) {
            out[d] = if (lrs[i] == 1) rrs[i] else lrs[i];
            d += 1;
        }
    }

    return out;
}

pub fn isBroadcastUnableComp(
    comptime spec1: anytype,
    comptime spec2: anytype,
) bool {
    const N1 = comptime dimsLen(spec1);
    const N2 = comptime dimsLen(spec2);

    if (N1 > N2) @compileError("can't broadcast to narrower shape " ++ std.fmt.comptimePrint("{} > {}\n", .{ N1, N2 }));

    inline for (0..N2) |i| {
        const d1 = comptime getSpecDimWithIdx(spec1, N1 - 1 - i);
        const d2 = comptime getSpecDimWithIdx(spec2, N2 - 1 - i);

        if (d1) |dl| {
            if (dl > 1) {
                if (d2) |dr| {
                    // 只有左右shape此维度值都大于1且不等，这种情况可以判断无法进行广播
                    if (dl != dr) return true else continue; // 此维度大小一致时，需要继续判断
                } else {
                    // 不知道第二个shape当前维度大小，实际判断需运行时决定，编译期检查继续
                    continue;
                }
            } else if (dl == 1) {
                // 此时这个维度肯定可以广播
                continue;
            } else {
                @compileError("invalid dimension");
            }
        } else {
            continue;
        }
    }

    return false;
}

pub fn getSpecDimWithName(comptime spec: anytype, comptime name: []const u8) ?usize {
    const info = @typeInfo(@TypeOf(spec)).@"struct";

    inline for (info.fields) |f| {
        if (comptime !std.mem.eql(u8, f.name, name)) continue;

        const td = f.defaultValue() orelse @compileError("Missing default value");

        if (f.type == usize) {
            return td;
        } else {
            return null;
        }
    }

    return null;
}

pub fn getSpecDimWithIdx(comptime spec: anytype, comptime idx: usize) ?usize {
    const info = @typeInfo(@TypeOf(spec)).@"struct";

    inline for (info.fields, 0..) |f, i| {
        if (idx != i) continue;

        const td = f.defaultValue() orelse @compileError("Missing default value");
        // @compileLog(std.fmt.comptimePrint("td: {any} f_type: {s}\n", .{ td, @typeName(f.type) }));

        if (f.type == usize or f.type == comptime_int) {
            return td;
        } else {
            return null;
        }
    }

    return null;
}

pub fn TypeShape(comptime dims: anytype) type {
    const info = @typeInfo(@TypeOf(dims)).@"struct";

    var fields: [info.fields.len]std.builtin.Type.StructField = undefined;
    // var dim_values: [info.fields.len]TypeDim = undefined;

    inline for (info.fields, 0..) |f, i| {
        // @compileLog("field name: " ++ f.name);
        const name = if (f.name.len == 0)
            std.fmt.comptimePrint("|_{d}", .{i})
        else
            f.name;

        const raw = @field(dims, f.name);
        const T = @TypeOf(raw);
        const ti = @typeInfo(T);

        // @compileLog(std.fmt.comptimePrint("ti: {s}\n", .{@typeName(T)}));

        const td: std.builtin.Type.StructField = switch (ti) {
            .int, .comptime_int => blk: {
                const value: usize = raw;
                break :blk std.builtin.Type.StructField{
                    .name = name,
                    .type = usize,
                    .default_value_ptr = @as(*const anyopaque, @ptrCast(&value)),
                    .is_comptime = false,
                    .alignment = @alignOf(usize),
                };
            },
            .pointer => |_| std.builtin.Type.StructField{
                .name = name,
                .type = @TypeOf(raw),
                .default_value_ptr = @as(*const anyopaque, @ptrCast(&raw)),
                .is_comptime = false,
                .alignment = @alignOf(T),
            },
            else => @compileError("shape: field '" ++ f.name ++ "' must be usize or string"),
        };

        // dim_values[i] = td;
        fields[i] = td;
    }

    return @Type(.{
        .@"struct" = .{
            .layout = .auto,
            .fields = &fields,
            .decls = &.{},
            .is_tuple = false,
        },
    });
}

pub fn isShapeSpec(comptime spec: anytype) bool {
    const ti = @typeInfo(@TypeOf(spec));
    switch (ti) {
        .@"struct" => |si| {
            for (si.fields) |f| {
                if (f.type != usize and f.type != comptime_int and !isString(f.type)) {
                    return false;
                }
            }

            return true;
        },
        else => return false,
    }
}

test "broadcast_comp" {
    const a =
        .{
            1,
            3,
            "H",
            "W",
        };

    const b = .{
        .N = 1,
        .C = 3,
        .H = "Height",
        .W = "Width",
    };

    const res = isBroadcastUnableComp(a, b);
    std.debug.print("res: {}\n", .{res});
}

test "shape creation" {
    const s1 = "ddddd";
    // var s2 = "dddddd";
    const res = isString(s1);
    std.debug.print("res: {}\n", .{res});

    const a =
        .{
            1,
            3,
            "H",
            "W",
        };
    // const TS1 = TypeShape(a);

    // inline for (@typeInfo(TS1).@"struct".fields) |f| {
    //     if (f.is_comptime) {
    //         std.debug.print("Field '{s}' is comptime\n", .{f.name});
    //     } else {
    //         std.debug.print("Field '{s}' is runtime\n", .{f.name});
    //     }
    // }
    //
    const b = .{
        .N = 1,
        .C = 3,
        .H = "Height",
        .W = "Width",
    };

    const TS = TypeShape(b);

    std.debug.print("a: {any} b: {} ts: {any}\n", .{ a, b, TS });

    const dict = &.{
        DictEntry{ .name = "H", .value = 224 },
        DictEntry{ .name = "W", .value = 227 },
    };

    const arr = evalShape(@TypeOf(a), dict);
    std.debug.print("arr: {any}\n", .{arr});
    // const arr1 = evalRuntimeShape(TS1, &.{
    //     .{ .name = "H", .value = 22 },
    //     .{ .name = "W", .value = 23 },
    // });

    // std.debug.print("{any} {any} {any}\n", .{ arr, arr1, TS1 });

    std.debug.print("idx: {?} name: {?}\n", .{ getSpecDimWithIdx(a, 1), getSpecDimWithName(a, "2") });
}
