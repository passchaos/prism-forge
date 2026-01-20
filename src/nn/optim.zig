const std = @import("std");
const tensor = @import("../tensor.zig");

pub const Param = struct {
    data: anyopaque,
    grad: anyopaque,
};

pub fn Optimizer(comptime T: type) type {
    return union(enum) {
        SGD: Sgd(T),
        MOMENTUM: Momentum(T),

        const Self = @This();

        pub fn update(
            self: *Self,
            params: []tensor.TensorView(T),
            grads: []const tensor.TensorView(T),
        ) !void {
            switch (self.*) {
                .SGD => |*sgd| try sgd.update(params, grads),
                .MOMENTUM => |*momentum| try momentum.update(params, grads),
            }
        }
    };
}

pub fn Sgd(comptime T: type) type {
    return struct {
        const Self = @This();

        lr: T,

        pub fn init(lr: T) Self {
            return Self{
                .lr = lr,
            };
        }

        pub fn update(self: *Self, params: []tensor.TensorView(T), grads: []const tensor.TensorView(T)) !void {
            for (params, grads) |*param, grad| {
                try param.addSubFused_(1.0, &grad, -1.0 * self.lr);
            }
        }
    };
}

pub fn Momentum(comptime T: type) type {
    return struct {
        const Self = @This();

        lr: T,
        momentum: T,
        velocity: ?[]tensor.TensorView(T) = null,
        allocator: std.mem.Allocator,

        pub fn init(lr: T, momentum: T, allocator: std.mem.Allocator) Self {
            return Self{ .lr = lr, .momentum = momentum, .allocator = allocator };
        }

        pub fn update(
            self: *Self,
            params: []tensor.TensorView(T),
            grads: []const tensor.TensorView(T),
        ) !void {
            if (self.velocity == null) {
                self.velocity = try self.allocator.alloc(tensor.TensorView(T), params.len);

                for (self.velocity.?, params) |*v, p| {
                    var new_v = try p.clone();
                    new_v.resetData(0);

                    v.* = new_v;
                }
            }

            for (params, grads, self.velocity.?) |*param, grad, *vel| {
                try vel.addSubFused_(self.momentum, &grad, -1.0 * self.lr);
                try param.add_(vel);
            }
        }
    };
}
