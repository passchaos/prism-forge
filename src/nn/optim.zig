const std = @import("std");

pub const Param = struct {
    data: anyopaque,
    grad: anyopaque,
};

pub fn Optimizer(comptime T: type) type {
    return union(enum) {
        SGD: Sgd(T),
        MOMENTUM: Momentum(T),

        const Self = @This();

        pub fn update(self: *Self, params: []const []T, grads: []const []const T) !void {
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

        pub fn update(self: *Self, params: []const []T, grads: []const []const T) !void {
            for (params, grads) |param, grad| {
                for (param, grad) |*p, g| {
                    p.* -= g * self.lr;
                }
            }
        }
    };
}

pub fn Momentum(comptime T: type) type {
    return struct {
        const Self = @This();

        lr: T,
        momentum: T,
        velocity: ?[][]T = null,
        allocator: std.mem.Allocator,

        pub fn init(lr: T, momentum: T, allocator: std.mem.Allocator) Self {
            return Self{ .lr = lr, .momentum = momentum, .allocator = allocator };
        }

        pub fn update(self: *Self, params: []const []T, grads: []const []const T) !void {
            if (self.velocity == null) {
                self.velocity = try self.allocator.alloc([]T, params.len);

                for (self.velocity.?, params) |*v, p| {
                    v.* = try self.allocator.alloc(T, p.len);
                    for (v.*) |*v_i| {
                        v_i.* = 0;
                    }
                }
            }
            for (params, grads, self.velocity.?) |param, grad, vel| {
                for (param, grad, vel) |*p, g, *v| {
                    v.* = self.momentum * v.* - self.lr * g;
                    p.* += v.*;
                }
            }
        }
    };
}
