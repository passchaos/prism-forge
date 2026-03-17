pub const classic = @import("classic.zig");
pub const word2vec = @import("word2vec.zig");

const std = @import("std");

pub fn dbg(v: anytype) @TypeOf(v) {
    const debug_info = std.debug.getSelfDebugInfo() catch return v;
    const address = @returnAddress();
    const module = debug_info.getModuleForAddress(address) catch return v;
    const symbol_info = module.getSymbolAtAddress(debug_info.allocator, address) catch return v;
    // defer symbol_info.deinit(debug_info.allocator);
    // std.debug.printSourceAtAddress(debug_info, writer: *Writer, address: usize, tty_config: Config)

    const file_name, const line, const column = if (symbol_info.source_location) |sl|
        .{ sl.file_name, sl.line, sl.column }
    else
        .{ "", 0, 0 };

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const cwd_path = std.fs.cwd().realpathAlloc(gpa.allocator(), ".") catch unreachable;

    if (std.mem.startsWith(u8, file_name, cwd_path)) {
        std.debug.print("begin with: {s}\n", .{cwd_path});
    }

    const relative_path = file_name[cwd_path.len..];
    // const relative_path = std.mem.trimStart(u8, file_name, cwd_path);

    const relative_file = std.mem.trimStart(u8, relative_path, "/");
    std.debug.print("cwd path: {s} {s}\n", .{ cwd_path, relative_file });

    std.debug.print("[{s}:{s} {s}:{d}:{d}] -> {any}", .{ symbol_info.name, symbol_info.compile_unit_name, relative_file, line, column, v });

    return v;
}

test "haha" {
    const a: usize = 1;
    const b = dbg(a + 1);
    std.debug.print("{d}\n", .{b});
}

test {
    @import("std").testing.refAllDecls(@This());
}
