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
        ADAGRAD: AdaGrad(T),
        ADAM: Adam(T),

        const Self = @This();

        pub fn deinit(self: *Self) void {
            switch (self.*) {
                .MOMENTUM => |*momentum| momentum.deinit(),
                .ADAGRAD => |*adagrad| adagrad.deinit(),
                .ADAM => |*adam| adam.deinit(),
                else => {},
            }
        }

        pub fn update(
            self: *Self,
            params: []tensor.TensorView(T),
            grads: []const tensor.TensorView(T),
        ) !void {
            if (params.len != grads.len) {
                return error.InvalidInput;
            }

            switch (self.*) {
                .SGD => |*sgd| try sgd.update(params, grads),
                .MOMENTUM => |*momentum| try momentum.update(params, grads),
                .ADAGRAD => |*adagrad| try adagrad.update(params, grads),
                .ADAM => |*adam| try adam.update(params, grads),
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

        pub fn deinit(self: *Self) void {
            if (self.velocity) |vel| {
                for (vel) |*v| {
                    v.deinit();
                }

                self.allocator.free(vel);
            }
        }

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

pub fn AdaGrad(comptime T: type) type {
    return struct {
        lr: T,
        h: ?[]tensor.TensorView(T) = null,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            if (self.h) |hr| {
                for (hr) |*h| {
                    h.deinit();
                }

                self.allocator.free(hr);
            }
        }

        pub fn init(lr: T, allocator: std.mem.Allocator) Self {
            return Self{ .lr = lr, .allocator = allocator };
        }

        pub fn update(
            self: *Self,
            params: []tensor.TensorView(T),
            grads: []const tensor.TensorView(T),
        ) !void {
            if (self.h == null) {
                self.h = try self.allocator.alloc(tensor.TensorView(T), params.len);

                for (self.h.?, params) |*h, p| {
                    var new_h = try p.clone();
                    new_h.resetData(0);

                    h.* = new_h;
                }
            }

            for (params, grads, self.h.?) |*param, grad, *h| {
                var grad_c = try grad.clone();
                defer grad_c.deinit();

                try grad_c.mul_(&grad_c);
                try h.add_(&grad_c);

                var h_c = try h.clone();
                defer h_c.deinit();
                h_c.sqrt_();
                h_c.addScalar_(1e-7);

                var grad_c_1 = try grad.clone();
                defer grad_c_1.deinit();
                try grad_c_1.div_(&h_c);
                grad_c_1.mulScalar_(self.lr);

                // std.debug.print("sub: {} orig: {}\n", .{ grad_c_1.data[5], grad.data[5] });

                try param.sub_(&grad_c_1);
            }
        }
    };
}

pub fn Adam(comptime T: type) type {
    return struct {
        lr: T,
        beta1: T,
        beta2: T,
        iter: usize = 0,
        allocator: std.mem.Allocator,
        m: ?[]tensor.TensorView(T) = null,
        v: ?[]tensor.TensorView(T) = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            if (self.m) |m| {
                for (m) |*mv| {
                    mv.deinit();
                }

                self.allocator.free(m);
            }
            if (self.v) |v| {
                for (v) |*vv| {
                    vv.deinit();
                }

                self.allocator.free(v);
            }
        }

        pub fn init(lr: T, beta1: T, beta2: T, allocator: std.mem.Allocator) Self {
            return Self{
                .lr = lr,
                .beta1 = beta1,
                .beta2 = beta2,
                .allocator = allocator,
            };
        }

        pub fn update(
            self: *Self,
            params: []tensor.TensorView(T),
            grads: []const tensor.TensorView(T),
        ) !void {
            if (self.m == null) {
                self.m = try self.allocator.alloc(tensor.TensorView(T), params.len);
                self.v = try self.allocator.alloc(tensor.TensorView(T), params.len);
                for (self.m.?, self.v.?, params) |*m, *v, p| {
                    var new_h = try p.clone();
                    new_h.resetData(0);

                    m.* = try new_h.clone();
                    v.* = new_h;
                }
            }

            self.iter += 1;

            const lr_t = self.lr * @sqrt(1.0 - std.math.pow(
                T,
                self.beta2,
                @as(T, @floatFromInt(self.iter)),
            )) / (1.0 - std.math.pow(
                T,
                self.beta1,
                @as(T, @floatFromInt(self.iter)),
            ));

            for (params, grads, self.m.?, self.v.?) |*param, grad, *m, *v| {
                var grad_m_c = try grad.clone();
                defer grad_m_c.deinit();

                try grad_m_c.sub_(m);
                grad_m_c.mulScalar_(1.0 - self.beta1);
                try m.add_(&grad_m_c);

                var grad_v_c = try grad.clone();
                defer grad_v_c.deinit();

                try grad_v_c.mul_(&grad_v_c);
                try grad_v_c.sub_(v);
                grad_v_c.mulScalar_(1.0 - self.beta2);
                try v.add_(&grad_v_c);

                var v_c = try v.clone();
                defer v_c.deinit();

                v_c.sqrt_();
                v_c.addScalar_(1e-7);

                var m_c = try m.clone();
                defer m_c.deinit();

                try m_c.div_(&v_c);
                m_c.mulScalar_(lr_t);

                try param.sub_(&m_c);
            }
        }
    };
}
