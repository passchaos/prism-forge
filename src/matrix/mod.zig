pub const svd = @import("svd.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
