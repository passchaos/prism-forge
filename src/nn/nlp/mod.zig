pub const classic = @import("classic.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
