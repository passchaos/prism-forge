const std = @import("std");
const utils = @import("utils.zig");

// const TypeDim = union(enum) {
//     static: usize,
//     named: []const u8,
// };

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

fn getSymbol(comptime dim: anytype) []const u8 {
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

pub fn evalRuntimeShape(comptime S: type, comptime dict: []const DictEntry) [@typeInfo(S).@"struct".fields.len]usize {
    const info = @typeInfo(S).@"struct";
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

pub fn dimsLen(comptime dims: anytype) usize {
    return @typeInfo(@TypeOf(dims)).@"struct".fields.len;
}

pub fn allStatic(comptime dims: anytype) bool {
    return staticDimLen(dims) == @typeInfo(@TypeOf(dims)).@"struct".fields.len;
}

pub fn staticDimLen(comptime dims: anytype) usize {
    const info = @typeInfo(@TypeOf(dims)).@"struct";

    var count: usize = 0;
    inline for (info.fields) |f| {
        if (f.type == usize or f.type == comptime_int) count += 1;
    }

    return count;
}

pub fn staticDims(comptime dims: anytype) [staticDimLen(dims)]usize {
    const info = @typeInfo(@TypeOf(dims)).@"struct";

    var out: [staticDimLen(dims)]usize = undefined;

    var d: usize = 0;
    inline for (info.fields, 0..) |f, i| {
        if (f.type == usize or f.type == comptime_int) {
            out[d] = i;
            d += 1;
        }
    }

    return out;
}

pub fn isBroadcastableComp(
    comptime spec1: anytype,
    comptime spec2: anytype,
) bool {
    const N1 = dimsLen(spec1);
    const N2 = dimsLen(spec2);

    if (N1 > N2) @compileError("can't broadcast to narrower shape");

    for (0..N2) |i| {
        const d1 = getDimCompWithIdx(spec1, N1 - 1 - i);
        const d2 = getDimCompWithIdx(spec2, N2 - 1 - i);
        
        if (d1) |dl| {
            
        }

        if (d1 == null or d2 == null) return false;
        if (d1 != d2) return false;
    }
}

pub fn getDimCompWithName(comptime dims: anytype, comptime name: []const u8) ?usize {
    const info = @typeInfo(@TypeOf(dims)).@"struct";

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

pub fn getDimCompWithIdx(comptime dims: anytype, comptime idx: usize) ?usize {
    const info = @typeInfo(@TypeOf(dims)).@"struct";

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

pub fn isTypeShapeSpec(comptime spec: anytype) bool {
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

    const arr = evalRuntimeShape(@TypeOf(a), dict);
    std.debug.print("arr: {any}\n", .{arr});
    // const arr1 = evalRuntimeShape(TS1, &.{
    //     .{ .name = "H", .value = 22 },
    //     .{ .name = "W", .value = 23 },
    // });

    // std.debug.print("{any} {any} {any}\n", .{ arr, arr1, TS1 });

    std.debug.print("idx: {?} name: {?}\n", .{ getDimCompWithIdx(a, 1), getDimCompWithName(a, "2") });
}
