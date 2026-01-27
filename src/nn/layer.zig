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

            // log.print(@src(), "init affine layout: w= {f} b= {f}\n", .{ w.layout, b.layout });

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
            // std.debug.print("backward: dout layout= {f}\n", .{dout.layout});
            const w_t = self.w.transpose();
            defer w_t.deinit();

            // log.print(@src(), "dout: dout: {f} dout_s: {any} size: {}\n", .{ dout.layout, @TypeOf(dout.*).S, dout.size() });
            // log.print(@src(), "dout: w_t: {f} w_t_s: {any} size: {}\n", .{ w_t.layout, @TypeOf(w_t).S, w_t.size() });

            // log.print(@src(), "dout: w: {f} w_s: {any} size: {}\n", .{ self.w.layout, @TypeOf(self.w).S, self.w.size() });
            const dx = try dout.matmul(&w_t);
            // std.debug.print("dx: {f}\n", .{dx});
            // std.debug.print("backward: self_x layout= {f}\n", .{self.x.?.layout});

            const x_t = self.x.?.transpose();
            defer x_t.deinit();

            // std.debug.print("x_t= {f} dout= {f}\n", .{ x_t.layout, dout.layout });
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

            // std.debug.print("backward: dw layout= {f}\n", .{n_dw.layout});

            return dx;
        }
    };
}

pub fn Dropout(comptime shape_expr_a: []const SizeExpr, comptime T: type) type {
    const Tensor = tensor.Tensor(shape_expr_a, T);

    return struct {
        const Self = @This();

        dropout_ratio: f32,
        mask: ?tensor.Tensor(shape_expr_a, T),

        pub fn init(dropout_ratio: f32) Self {
            return Self{
                .dropout_ratio = dropout_ratio,
                .mask = null,
            };
        }

        pub fn forward(self: *Self, x: *const Tensor) !Tensor {
            if (self.mask == null) {
                var mask = try tensor.rand(
                    x.s_allocator(),
                    shape_expr_a,
                    x.layout.shape_env(),
                    @as(T, 0.0),
                    1.0,
                );

                const func = struct {
                    fn call(input: T, ctx: f32) T {
                        return if (input < ctx) @as(T, 0.0) else @as(T, 1.0);
                    }
                }.call;
                mask.map_(self.dropout_ratio, func);

                self.mask = mask;
            }

            return x.mul(&self.mask.?);
        }

        pub fn backward(self: *Self, dout: *const Tensor) !Tensor {
            return try dout.mul(&self.mask.?);
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

pub fn Convolution(
    comptime N: SizeExpr,
    comptime C: SizeExpr,
    comptime H: SizeExpr,
    comptime W: SizeExpr,
    comptime FN: SizeExpr,
    comptime FH: SizeExpr,
    comptime FW: SizeExpr,
    comptime pads: [4]SizeExpr,
    comptime stride: SizeExpr,
    comptime T: type,
) type {
    return struct {
        const OH = H.add(pads[2].add(pads[3])).sub(&FH).div(&stride).add(SizeExpr.static(1));
        const OW = W.add(pads[0].add(pads[1])).sub(&FW).div(&stride).add(SizeExpr.static(1));

        const WT = tensor.Tensor(&.{ FN, C, FH, FW }, T);
        const BT = tensor.Tensor(&.{ FN, SizeExpr.static(1), SizeExpr.static(1) }, T);
        const IT = tensor.Tensor(&.{ N, C, H, W }, T);
        const OT = tensor.Tensor(&.{ N, FN, OH, OW }, T);
        const Self = @This();

        allocator: std.mem.Allocator,
        w: WT,
        b: BT,
        shape_env: *const shape_expr.ShapeEnv,

        pub fn deinit(self: *Self) void {
            self.w.deinit();
            self.b.deinit();
        }

        fn filterImpl(
            self: *const Self,
            data: *const tensor.Tensor(&.{ N, C, H, W }, T),
            shape_env: *const shape_expr.ShapeEnv,
        ) !tensor.Tensor(&.{ N, FN, OH, OW }, T) {
            const padded_data = try data.pad(&pads, @as(T, 0.0));
            defer padded_data.deinit();

            const n_v = try N.eval(shape_env);
            const fn_v = try FN.eval(shape_env);
            const c_v = try C.eval(shape_env);
            const fh_v = try FH.eval(shape_env);
            const fw_v = try FW.eval(shape_env);
            const oh_v = try OH.eval(shape_env);
            const ow_v = try OW.eval(shape_env);

            var result = try tensor.zeros(self.allocator, T, &.{ N, FN, OH, OW }, shape_env);

            for (0..n_v) |n_i| {
                for (0..fn_v) |fn_i| {
                    for (0..c_v) |c_i| {
                        for (0..oh_v) |oh_i| {
                            for (0..ow_v) |ow_i| {
                                const r_idx = [4]usize{ n_i, fn_i, oh_i, ow_i };

                                for (0..fh_v) |fh_i| {
                                    for (0..fw_v) |fw_i| {
                                        const data_idx = [4]usize{ n_i, c_i, oh_i + fh_i, ow_i + fw_i };
                                        const filter_idx = [4]usize{ fn_i, c_i, fh_i, fw_i };

                                        const d_v = try padded_data.getData(data_idx);
                                        const f_v = try self.w.getData(filter_idx);

                                        const orig_v = try result.getData(r_idx);

                                        try result.setData(r_idx, orig_v + d_v * f_v);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        pub fn init(allocator: std.mem.Allocator, shape_env: *const shape_expr.ShapeEnv, weight_init_std: T) !Self {
            var w = try tensor.randNorm(allocator, &.{ FN, C, FH, FW }, shape_env, weight_init_std, 1.0);
            w.mulScalar_(weight_init_std);

            const b = try tensor.zeros(allocator, T, &.{ FN, SizeExpr.static(1), SizeExpr.static(1) }, shape_env);

            return Self.initImpl(allocator, shape_env, w, b);
        }

        pub fn initImpl(
            allocator: std.mem.Allocator,
            shape_env: *const shape_expr.ShapeEnv,
            w: tensor.Tensor(&.{ FN, C, FH, FW }, T),
            b: tensor.Tensor(&.{ FN, SizeExpr.static(1), SizeExpr.static(1) }, T),
        ) Self {
            return Self{
                .allocator = allocator,
                .w = w,
                .b = b,
                .shape_env = shape_env,
            };
        }

        pub fn forward(self: *Self, x: *const IT) !OT {
            var result = try self.filterImpl(x, self.shape_env);

            const b_b = self.b.broadcastTo(&.{ N, FN, OH, OW });
            defer b_b.deinit();

            result.add_(&b_b);

            return result;
        }

        pub fn backward(self: *Self, dout: *const OT) !IT {
            _ = self;
            _ = dout;
            return error.NoImpl;
        }
    };
}

test "Convolution" {
    const allocator = std.testing.allocator;

    {
        const N = comptime SizeExpr.sym(.{ .name = "batch_size" });
        const C = comptime SizeExpr.static(1);
        const H = comptime SizeExpr.static(4);
        const W = comptime SizeExpr.static(4);

        const FN = comptime SizeExpr.sym(.{ .name = "filter_num" });
        const FH = comptime SizeExpr.static(3);
        const FW = comptime SizeExpr.static(3);

        const PAD = comptime SizeExpr.sym(.{ .name = "pad" });
        const STRIDE = comptime SizeExpr.static(1);

        std.debug.print(
            "N= {f} C= {f} H= {f} W= {f} FN= {f} FH= {f} FW= {f}\n",
            .{ N, C, H, W, FN, FH, FW },
        );

        const Conv = Convolution(
            N,
            C,
            H,
            W,
            FN,
            FH,
            FW,
            [4]SizeExpr{ PAD, PAD, PAD, PAD },
            STRIDE,
            f32,
        );
        std.debug.print("OH= {f} OW= {f}\n", .{ Conv.OH, Conv.OW });

        var shape_env = try shape_expr.ShapeEnv.init(allocator);
        defer shape_env.deinit();

        try shape_env.bind(&N.Sym, 1);
        try shape_env.bind(&FN.Sym, 1);
        try shape_env.bind(&PAD.Sym, 1);

        var w = try tensor.fromArray(allocator, [4][4]f32{
            [4]f32{ 1.0, 2.0, 3.0, 0.0 },
            [4]f32{ 0.0, 1.0, 2.0, 3.0 },
            [4]f32{ 3.0, 0.0, 1.0, 2.0 },
            [4]f32{ 2.0, 3.0, 0.0, 1.0 },
        }, &shape_env);
        defer w.deinit();

        var w_f = w.reshape(&.{ N, C, H, W });
        defer w_f.deinit();

        var filter = try tensor.fromArray(allocator, [3][3]f32{
            [3]f32{ 2.0, 0.0, 1.0 },
            [3]f32{ 0.0, 1.0, 2.0 },
            [3]f32{ 1.0, 0.0, 2.0 },
        }, &shape_env);
        defer filter.deinit();

        const filter_f = filter.reshape(&.{ FN, C, FH, FW });

        const b = try tensor.full(allocator, &.{ FN, SizeExpr.static(1), SizeExpr.static(1) }, &shape_env, @as(f32, 3.0));

        var conv_layer = Conv.initImpl(allocator, &shape_env, filter_f, b);
        defer conv_layer.deinit();

        var result = try conv_layer.forward(&w_f);
        defer result.deinit();

        try std.testing.expectEqual(10, try result.getData([4]usize{ 0, 0, 0, 0 }));
        try std.testing.expectEqual(5, try result.getData([4]usize{ 0, 0, 0, 3 }));

        try std.testing.expectEqual(7, try result.getData([4]usize{ 0, 0, 1, 0 }));
        try std.testing.expectEqual(9, try result.getData([4]usize{ 0, 0, 2, 3 }));

        try std.testing.expectEqual(7, try result.getData([4]usize{ 0, 0, 3, 2 }));
        try std.testing.expectEqual(6, try result.getData([4]usize{ 0, 0, 3, 3 }));

        std.debug.print("result: {f}\n", .{result});
    }
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
