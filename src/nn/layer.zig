const std = @import("std");
const tensor = @import("../tensor.zig");
const log = @import("../log.zig");

pub fn Relu(comptime N: usize, comptime T: type) type {
    const Tensor = tensor.Tensor(N, .{ .T = T });
    const BoolTensor = tensor.Tensor(N, .{ .T = bool });

    return struct {
        mask: ?BoolTensor = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.mask.?.deinit();
        }

        pub fn init() Self {
            return Self{};
        }

        pub fn forward(self: *Self, x: *Tensor) !void {
            if (self.mask) |m_i| {
                m_i.deinit();
            }
            self.mask = try x.le(@as(T, 0));
            // log.print(@src(), "mask layout: {f}\n", .{self.mask.?.layout});

            try x.maskFill_(self.mask.?, @as(T, 0));
        }

        pub fn backward(self: *Self, dout: *Tensor) !void {
            try dout.maskFill_(self.mask.?, @as(T, 0));
        }
    };
}

pub fn Sigmoid(comptime N: usize, comptime T: type) type {
    const Tensor = tensor.Tensor(N, .{ .T = T });

    return struct {
        out: ?Tensor,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.out.?.deinit();
        }

        pub fn init() Sigmoid {
            return Sigmoid{};
        }

        pub fn forward(self: *Self, x: *const Tensor) !Tensor {
            x.sigmoid_();

            self.out = x.*;

            return x;
        }

        pub fn backward(self: *Self, dout: *const Tensor) !Tensor {
            var tmp = try dout.clone();
            try tmp
                .mul_(@as(T, -1));

            try tmp.add_(@as(T, 1));
            try tmp.mul_(dout);
            try tmp.mul_(self.out.?);

            return tmp;
        }
    };
}

pub fn Affine(comptime N: usize, comptime T: type) type {
    const Tensor = tensor.Tensor(N, .{ .T = T });

    return struct {
        w: Tensor,
        b: Tensor,
        x: ?Tensor = null,
        dw: ?Tensor = null,
        db: ?Tensor = null,

        const Self = @This();

        pub fn take_dinfo(self: *Self) struct { dw: ?Tensor, db: ?Tensor } {
            const dw_r = self.dw;
            const dw_b = self.db;

            self.dw = null;
            self.db = null;

            return .{
                .dw = dw_r,
                .db = dw_b,
            };
        }

        pub fn deinit(self: *Self) void {
            self.w.deinit();

            self.b.deinit();

            if (self.x) |x_r| {
                x_r.deinit();
            }

            if (self.dw) |dw_r| {
                dw_r.deinit();
            }

            if (self.db) |db_r| {
                db_r.deinit();
            }
        }

        pub fn init(w: Tensor, b: Tensor) Self {
            return Self{
                .w = w,
                .b = b,
            };
        }

        pub fn forward(self: *Self, x: *const Tensor) !Tensor {
            const x_c = try x.clone();

            if (self.x) |x_r| {
                x_r.deinit();
            }
            self.x = x_c;

            var out = try x.matmul(&self.w);
            try out.add_(&self.b);

            return out;
        }

        pub fn backward(self: *Self, dout: *const Tensor) !Tensor {
            const w_t = self.w.transpose();
            defer w_t.deinit();

            // std.debug.print("dout: {f} w_t: {f}\n", .{ dout, w_t });
            const dx = try dout.matmul(&w_t);
            // std.debug.print("dx: {f}\n", .{dx});

            const x_t = self.x.?.transpose();
            defer x_t.deinit();

            const n_dw = try x_t.matmul(dout);
            const n_db = try dout.sum(0);

            if (self.dw) |dwr| {
                dwr.deinit();
            }
            if (self.db) |dbr| {
                dbr.deinit();
            }

            self.dw = n_dw;
            self.db = n_db;

            return dx;
        }
    };
}

pub fn SoftmaxWithLoss(comptime N: usize, comptime T: type) type {
    const Tensor = tensor.Tensor(N, .{ .T = T });
    const Tensor0 = tensor.Tensor(0, .{ .T = T });

    return struct {
        y: ?Tensor = null,
        t: ?Tensor = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.y.?.deinit();
            self.t.?.deinit();
        }

        pub fn init() Self {
            return Self{};
        }

        pub fn forward(self: *Self, x: *const Tensor, t: *const Tensor) !Tensor0 {
            if (self.y) |y_r| {
                y_r.deinit();
            }
            if (self.t) |t_r| {
                t_r.deinit();
            }

            self.y = try x.softmax();
            // std.debug.print("self_y: {f}\n", .{self.y.?});
            self.t = try t.clone();
            // log.print(@src(), "self_t: {f}\n", .{self.t.?});

            const loss = try self.y.?.crossEntropy(&self.t.?);

            return loss;
        }

        pub fn backward(self: *Self) !Tensor {
            const batch_size = self.t.?.shape()[0];

            // std.debug.print("y: {f} t: {f}\n", .{ self.y.?, self.t.? });
            var dx = try self.y.?.sub(self.t.?);
            try dx.div_(@as(T, @floatFromInt(batch_size)));

            return dx;
        }
    };
}

// pub fn WeightInit(comptime T: type) type {
//     return union(enum) { std: T, xavier, he };
// }

// pub const Linear = struct {
//     allocator: std.mem.Allocator,
//     _weight: Tensor,
//     bias: ?Tensor,
//     x: ?Tensor,
//     x_original_shape: ?[]const usize,

//     const Self = @This();

//     pub fn new(allocator: std.mem.Allocator, comptime T: type, weight_init: WeightInit(T), input_size: usize, output_size: usize) !Self {
//         switch (@typeInfo(T)) {
//             inline .float => {
//                 var weights = try Tensor.randNorm(allocator, T, &.{ input_size, output_size }, 0.0, 1.0);

//                 const scale = switch (weight_init) {
//                     .std => |std_v| std_v,
//                     .xavier => 6.0 / @as(T, @floatFromInt(input_size + output_size)),
//                     .he => 2.0 / @sqrt(input_size),
//                 };

//                 try weights.mul_(scale);

//                 const bias = try Tensor.zeros(allocator, T, &.{output_size});

//                 return Self{
//                     .allocator = allocator,
//                     ._weight = weights,
//                     .bias = bias,
//                     .x = null,
//                     .x_original_shape = null,
//                 };
//             },
//             inline else => @compileError("Unsupported type" ++ @typeName(T)),
//         }
//     }

//     pub fn forward(self: *Self, input: Tensor) !Tensor {
//         self.x_original_shape = try self.allocator.dupe(usize, input.shapes());

//         self.x = try input.clone();

//         const out = if (self.bias) |bias|
//             try (try input.matmul(&self._weight)).add(bias)
//         else
//             try input.matmul(&self._weight);
//         return out;
//     }

//     pub fn backward(self: *Self, grad: Tensor) !Tensor {
//         var wi = try self._weight.clone();
//         try wi.transpose_();

//         const dx = try grad.matmul(&wi);

//         var x_t = try self.x.?.clone();
//         try x_t.transpose_();

//         // const dw = try x_t.matmul(&grad);

//         return dx;
//     }
// };
const Tensor2 = tensor.Tensor(2, .{ .T = f64 });
const TestNet = struct {
    const Self = @This();

    w: Tensor2,
    b: Tensor2,

    fn deinit(self: *Self) void {
        self.w.deinit();
        self.b.deinit();
    }

    fn init(w: Tensor2, b: Tensor2) Self {
        return Self{
            .w = w,
            .b = b,
        };
    }

    pub fn loss(self: *Self, x: *const Tensor2, t: *const Tensor2) !f64 {
        var f1 = try x.matmul(&self.w);
        defer f1.deinit();

        try f1.add_(&self.b);

        const f2 = try f1.softmax();
        defer f2.deinit();
        const f3 = try f2.crossEntropy(t);
        defer f3.deinit();

        const res = try f3.dataItem();
        return res;
    }
};
test "numerical_gradient" {
    const allocator = std.testing.allocator;

    const w = try tensor.randNorm(allocator, [2]usize{ 2, 3 }, 0.0, 1.0);
    const b = try tensor.rand(allocator, [2]usize{ 1, 3 }, 0.0, 1.0);

    const input = try tensor.rand(allocator, [2]usize{ 2, 2 }, 5.0, 10.0);
    defer input.deinit();
    const t = try tensor.fromArray(allocator, [2][3]f64{
        .{ 0.0, 0.0, 1.0 },
        .{ 1.0, 0.0, 0.0 },
    });
    defer t.deinit();

    const layer_res = blk: {
        const AffineT = Affine(2, f64);
        const SoftmaxWithLossT = SoftmaxWithLoss(2, f64);

        var aff = AffineT.init(try w.clone(), try b.clone());
        defer aff.deinit();
        var swl = SoftmaxWithLossT.init();
        defer swl.deinit();

        const f1 = try aff.forward(&input);
        defer f1.deinit();
        const f2 = try swl.forward(&f1, &t);
        defer f2.deinit();

        const layer_loss = try f2.dataItem();
        // std.debug.print("layer loss: {f}\n", .{f2});

        const b2 = try swl.backward();
        defer b2.deinit();
        const b1 = try aff.backward(&b2);
        defer b1.deinit();

        // std.debug.print("dw: {f}\ndb: {f}\n", .{ aff.dw.?, aff.db.? });
        break :blk .{ layer_loss, try aff.dw.?.clone(), try aff.db.?.clone() };
    };
    defer layer_res.@"1".deinit();
    defer layer_res.@"2".deinit();

    const numerical_res = blk: {
        const basic = @import("basic.zig");

        var test_net = TestNet.init(w, b);
        defer test_net.deinit();

        const loss = try test_net.loss(&input, &t);
        // try std.testing.expectEqual(layer_loss, loss);

        // std.debug.print("numerical loss: {}\n", .{loss});

        const dw_o = try basic.numericalGradient(allocator, &test_net, basic.LossArgument{
            .x = &input,
            .t = &t,
        }, basic.net_loss, &test_net.w);
        // defer dw_o.deinit();

        const db_o = try basic.numericalGradient(allocator, &test_net, basic.LossArgument{
            .x = &input,
            .t = &t,
        }, basic.net_loss, &test_net.b);
        // defer db_o.deinit();

        break :blk .{ loss, dw_o, db_o };
    };
    defer numerical_res.@"1".deinit();
    defer numerical_res.@"2".deinit();

    try std.testing.expectEqual(layer_res.@"0", numerical_res.@"0");
    std.debug.print("dw:\nlayer= {f}\nnumerical= {f}\n", .{ layer_res.@"1", numerical_res.@"1" });
}

test "affine_relu" {
    const allocator = std.testing.allocator;

    const AffineT = Affine(2, f64);
    const ReluT = Relu(2, f64);
    const SoftWithLossT = SoftmaxWithLoss(2, f64);

    const w = try tensor.arange(allocator, f64, .{ .end = 10 });
    defer w.deinit();

    const w1 = try w.reshape([2]usize{ 2, 5 });
    // defer w1.deinit();

    const ab = try tensor.arange(allocator, f64, .{ .end = 5 });
    defer ab.deinit();

    const ab1 = try ab.reshape([2]usize{ 1, 5 });
    // defer ab1.deinit();

    var affine = AffineT.init(w1, ab1);
    defer affine.deinit();

    var relu = ReluT.init();
    defer relu.deinit();

    var swl = SoftWithLossT.init();
    defer swl.deinit();

    const x = try tensor.arange(allocator, f64, .{ .end = 6 });
    defer x.deinit();

    const x1 = try x.reshape([2]usize{ 3, 2 });
    defer x1.deinit();

    const t = try tensor.fromArray(allocator, [3][5]f64{
        .{ 0, 0, 0, 0, 1 },
        .{ 0, 0, 0, 1, 0 },
        .{ 1, 0, 0, 0, 0 },
    });
    defer t.deinit();

    std.debug.print("input: {f}\n", .{x1});
    const f0 = try affine.forward(&x1);
    defer f0.deinit();

    const f1 = try relu.forward(&f0);
    defer f1.deinit();

    std.debug.print("f1: {f}\n", .{f1});

    const f2 = try swl.forward(&f1, &t);
    defer f2.deinit();

    std.debug.print("f2: {f}\n", .{f2});

    const b2 = try swl.backward();
    defer b2.deinit();
    std.debug.print("b2: {f}\n", .{b2});

    const b1 = try relu.backward(&b2);
    defer b1.deinit();

    std.debug.print("b1: {f}\n", .{b1});

    const b0 = try affine.backward(&b1);
    defer b0.deinit();

    std.debug.print("b0: {f}\n", .{b0});
    // try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, b0.shapes());
}
