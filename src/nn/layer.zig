const std = @import("std");
const tensor = @import("../tensor.zig");
const log = @import("../log.zig");
const shape_expr = @import("../shape_expr.zig");

const SizeExpr = shape_expr.SizeExpr;

pub const Layer = enum {
    Relu,
    Affine,
    SoftmaxWithLoss,
};

pub fn Relu(comptime shape: []const SizeExpr, comptime T: type) type {
    const Tensor = tensor.Tensor(shape, T);
    const BoolTensor = tensor.Tensor(shape, bool);

    return struct {
        tag: Layer = Layer.Relu,
        mask: ?BoolTensor = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.mask.?.deinit();
        }

        pub fn init() Self {
            return Self{};
        }

        pub fn forward(self: *Self, x: *const Tensor) !Tensor {
            if (self.mask) |m_i| {
                m_i.deinit();
            }
            self.mask = try x.leScalar(0);

            var nr = try x.clone();
            // log.print(@src(), "mask layout: {f}\n", .{self.mask.?.layout});

            try nr.maskFill_(self.mask.?, @as(T, 0));

            return nr;
        }

        pub fn backward(self: *Self, dout: *const Tensor) !Tensor {
            var res = try dout.clone();
            try res.maskFill_(self.mask.?, @as(T, 0));

            return res;
        }
    };
}

pub fn AffineWeight(comptime T: type) type {
    return union(enum) { Std: T, Xavier, He };
}

pub fn AffineWeightGradView(comptime T: type) type {
    return struct {
        w_view: tensor.TensorView(T),
        dw_view: tensor.TensorView(T),
        b_view: tensor.TensorView(T),
        db_view: tensor.TensorView(T),
    };
}

pub fn Affine(comptime batch_size: SizeExpr, comptime input_size: SizeExpr, comptime output_size: SizeExpr, comptime T: type) type {
    const Tensor = tensor.Tensor(&.{ batch_size, input_size }, T);
    const TensorW = tensor.Tensor(&.{ input_size, output_size }, T);
    const TensorB = tensor.Tensor(&.{ SizeExpr.static(1), output_size }, T);
    const TensorG = tensor.Tensor(&.{ batch_size, output_size }, T);

    return struct {
        tag: Layer = Layer.Affine,
        w: TensorW,
        b: TensorB,
        x: ?Tensor = null,
        dw: ?TensorW = null,
        db: ?TensorB = null,

        const Self = @This();

        pub fn take_dinfo(self: *Self) struct { dw: ?TensorW, db: ?TensorB } {
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

        pub fn init(allocator: std.mem.Allocator, shape_env: *const shape_expr.ShapeEnv, weight_init: AffineWeight(T)) !Self {
            var w = try tensor.randNorm(allocator, &.{ input_size, output_size }, shape_env, 0.0, 1.0);
            const scale = switch (weight_init) {
                .Std => |init_std| init_std,
                .Xavier => blk: {
                    const fan_in = try input_size.eval(shape_env);
                    const scale = @sqrt(2.0 / @as(T, @floatFromInt(fan_in)));
                    break :blk scale;
                },
                .He => blk: {
                    const fan_in = try input_size.eval(shape_env);
                    const scale = @sqrt(2.0 / @as(T, @floatFromInt(fan_in)));
                    break :blk scale;
                },
            };
            w.mulScalar_(scale);

            const b = try tensor.zeros(allocator, T, &.{ shape_expr.SizeExpr.static(1), output_size }, shape_env);

            return Self{
                .w = w,
                .b = b,
            };
        }

        pub fn forward(self: *Self, x: *const Tensor) !TensorG {
            const x_c = try x.clone();

            if (self.x) |x_r| {
                x_r.deinit();
            }
            self.x = x_c;

            var out = try x.matmul(&self.w);
            const broadcasted_b = self.b.broadcastTo(@TypeOf(out).S);
            defer broadcasted_b.deinit();

            out.add_(&broadcasted_b);

            return out;
        }

        pub fn backward(self: *Self, dout: *const TensorG) !Tensor {
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

        pub fn weightGradView(self: *const Self) !AffineWeightGradView(T) {
            const w_view = self.w.view();
            const dw_view = self.dw.?.view();
            const b_view = self.b.view();
            const db_view = self.db.?.view();

            return .{
                .w_view = w_view,
                .dw_view = dw_view,
                .b_view = b_view,
                .db_view = db_view,
            };
        }
    };
}

pub fn SoftmaxWithLoss(comptime shape: []const SizeExpr, comptime T: type) type {
    const Tensor = tensor.Tensor(shape, T);

    return struct {
        tag: Layer = Layer.SoftmaxWithLoss,
        y: ?Tensor = null,
        t: ?Tensor = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            if (self.y) |dw_r| {
                dw_r.deinit();
            }
            if (self.t) |db_r| {
                db_r.deinit();
            }
        }

        pub fn init() Self {
            return Self{};
        }

        pub fn forward(self: *Self, x: *const Tensor, t: *const Tensor) !T {
            if (self.y) |y_r| {
                y_r.deinit();
            }
            if (self.t) |t_r| {
                t_r.deinit();
            }

            // log.print(@src(), "x: {f} t: {f}\n", .{ x.layout, t.layout });
            const loss = try x.crossEntropyLogits(t);
            defer loss.deinit();

            self.y = try x.softmax();
            // std.debug.print("self_y: {f}\n", .{self.y.?});
            self.t = try t.clone();
            // log.print(@src(), "self_t: {f}\n", .{self.t.?});

            // const loss = try self.y.?.crossEntropy(&self.t.?);

            return try loss.dataItem();
        }

        pub fn backward(self: *Self) !Tensor {
            const batch_size = self.t.?.shape()[0];

            // std.debug.print("y: {f} t: {f}\n", .{ self.y.?, self.t.? });
            var dx = try self.y.?.sub(&self.t.?);
            dx.divScalar_(@as(T, @floatFromInt(batch_size)));

            return dx;
        }
    };
}

test "affine_relu" {
    const allocator = std.testing.allocator;

    const AffineT = Affine(
        SizeExpr.sym(.{ .name = "batch_size" }),
        SizeExpr.static(20),
        SizeExpr.static(50),
        f64,
    );
    const ReluT = Relu(&.{ SizeExpr.sym(.{ .name = "batch_size" }), SizeExpr.static(50) }, f64);
    const SoftWithLossT = SoftmaxWithLoss(&.{ SizeExpr.sym(.{ .name = "batch_size" }), SizeExpr.static(50) }, f64);

    const w = try tensor.arange(allocator, 10.0, .{});
    defer w.deinit();

    const w1 = try w.reshape(&.{ SizeExpr.static(2), SizeExpr.static(5) });
    // defer w1.deinit();

    const ab = try tensor.arange(allocator, 5.0, .{});
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
