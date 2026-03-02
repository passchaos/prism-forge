pub const conv_net = @import("conv_net.zig");
pub const mlp = @import("mlp.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
