pub const nlp = @import("nlp/mod.zig");
pub const optim = @import("optim.zig");
pub const model = @import("model/mod.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
