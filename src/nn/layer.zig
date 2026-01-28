const std = @import("std");
const tensor = @import("../tensor.zig");
const log = @import("../log.zig");
const shape_expr = @import("../shape_expr.zig");
const utils = @import("../utils.zig");
const tools = @import("tools.zig");

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
            const w_t = self.w.transpose();
            defer w_t.deinit();

            const dx = try dout.matmul(&w_t);

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
        const F_OH = H.add(pads[2]).add(pads[3]).sub(FH).div(stride).add(SizeExpr.static(1));
        const F_OW = W.add(pads[0]).add(pads[1]).sub(FW).div(stride).add(SizeExpr.static(1));

        const IM2COL_M = N.mul(F_OH).mul(F_OW);
        const IM2COL_K = FH.mul(FW).mul(C);

        const WT = tensor.Tensor(&.{ FN, C, FH, FW }, T);
        const BT = tensor.Tensor(&.{FN}, T);
        const IT = tensor.Tensor(&.{ N, C, H, W }, T);
        const OT = tensor.Tensor(&.{ N, FN, F_OH, F_OW }, T);

        const ImSE = utils.tensor.computePaddedShape(&.{ N, C, H, W }, &pads);
        const ImT = tensor.Tensor(&ImSE, T);

        const Self = @This();

        allocator: std.mem.Allocator,
        w: WT,
        b: BT,
        shape_env: *const shape_expr.ShapeEnv,

        dw: ?tensor.Tensor(&.{ FN, C, FH, FW }, T) = null,
        db: ?tensor.Tensor(&.{ SizeExpr.static(1), FN }, T) = null,

        x_col: ?tensor.Tensor(&.{ IM2COL_M, IM2COL_K }, T) = null,
        w_col: ?tensor.Tensor(&.{ IM2COL_K, FN }, T) = null,

        pub fn deinit(self: *Self) void {
            self.w.deinit();
            self.b.deinit();

            if (self.dw) |dw| {
                dw.deinit();
            }

            if (self.db) |db| {
                db.deinit();
            }

            if (self.x_col) |x_col| {
                x_col.deinit();
            }

            if (self.w_col) |w_col| {
                w_col.deinit();
            }
        }

        fn filterIm2ColImpl(
            self: *const Self,
            data: *const tensor.Tensor(&.{ N, C, H, W }, T),
            shape_env: *const shape_expr.ShapeEnv,
        ) !tensor.Tensor(&.{ N, FN, F_OH, F_OW }, T) {
            const cols = try tools.im2col(
                N,
                C,
                H,
                W,
                FH,
                FW,
                pads,
                stride,
                T,
                self.allocator,
                data,
                shape_env,
            );
            defer cols.deinit();
            std.debug.print("cols: {f}\n", .{cols});

            const filters = try self.w.reshape(&.{ FN, IM2COL_K });
            defer filters.deinit();
            const filters_w = filters.transpose();
            defer filters_w.deinit();

            const res1 = try cols.matmul(&filters_w);
            defer res1.deinit();
            const res2 = try res1.reshape(&.{ N, FN, F_OH, F_OW });
            std.debug.print("res: {f}\n", .{res2});

            return res2;
        }

        fn filterImpl(
            self: *const Self,
            data: *const tensor.Tensor(&.{ N, C, H, W }, T),
            shape_env: *const shape_expr.ShapeEnv,
        ) !tensor.Tensor(&.{ N, FN, F_OH, F_OW }, T) {
            const padded_data = try data.pad(&pads, @as(T, 0.0));
            defer padded_data.deinit();

            const n_v = try N.eval(shape_env);
            const fn_v = try FN.eval(shape_env);
            const c_v = try C.eval(shape_env);
            const fh_v = try FH.eval(shape_env);
            const fw_v = try FW.eval(shape_env);
            const oh_v = try F_OH.eval(shape_env);
            const ow_v = try F_OW.eval(shape_env);

            var result = try tensor.zeros(self.allocator, T, &.{ N, FN, F_OH, F_OW }, shape_env);

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
            var w = try tensor.randNorm(
                allocator,
                &.{ FN, C, FH, FW },
                shape_env,
                weight_init_std,
                1.0,
            );
            w.mulScalar_(weight_init_std);

            const b = try tensor.zeros(allocator, T, &.{FN}, shape_env);

            return Self.initImpl(allocator, shape_env, w, b);
        }

        pub fn initImpl(
            allocator: std.mem.Allocator,
            shape_env: *const shape_expr.ShapeEnv,
            w: tensor.Tensor(&.{ FN, C, FH, FW }, T),
            b: tensor.Tensor(&.{FN}, T),
        ) Self {
            return Self{
                .allocator = allocator,
                .w = w,
                .b = b,
                .shape_env = shape_env,
            };
        }

        pub fn forward(self: *Self, x: *const IT) !OT {
            const cols = try tools.im2col(
                N,
                C,
                H,
                W,
                FH,
                FW,
                pads,
                stride,
                T,
                self.allocator,
                x,
                self.shape_env,
            );

            const filters = try self.w.reshape(&.{ FN, IM2COL_K });
            defer filters.deinit();
            const filters_w = filters.transpose();

            // result shape: [N * F_OH * F_OW, FN]
            var res1 = try cols.matmul(&filters_w);
            defer res1.deinit();

            // const type_equal = comptime @TypeOf(cols).S[0].equal(IM2COL_M);
            // @compileLog(@TypeOf(cols).S[0]);
            // @compileLog(IM2COL_M);

            // comptime {
            // @setEvalBranchQuota(20000);
            // @compileLog("0 dim: " ++ std.fmt.comptimePrint("{f} {f}\n", .{ @TypeOf(cols).S[0], IM2COL_M }));
            // }
            self.x_col = cols;
            self.w_col = filters_w;

            var b_b = self.b.broadcastTo(@TypeOf(res1).S);
            defer b_b.deinit();

            res1.add_(&b_b);

            const result = try res1.reshape(&.{ N, F_OH, F_OW, FN });
            defer result.deinit();

            return result.permute([4]usize{ 0, 3, 1, 2 });
        }

        pub fn backward(self: *Self, dout: *const OT) !IT {
            const dout1 = dout.permute([4]usize{ 0, 2, 3, 1 });
            defer dout1.deinit();
            const dout2 = try dout1.reshape(&.{ N.mul(F_OH).mul(F_OW), FN });
            defer dout2.deinit();

            self.db = try dout2.sum(0);

            const x_col_t = self.x_col.?.transpose();
            defer x_col_t.deinit();

            // dw shape: [C.mul(FH).mul(FW), FN]
            const dw = try x_col_t.matmul(&dout2);
            defer dw.deinit();

            const dw_t = dw.transpose();
            defer dw_t.deinit();

            self.dw = dw_t.reshape(&.{ FN, C, FH, FW });

            const w_col_t = self.w_col.?.transpose();
            defer w_col_t.deinit();
            const dcol = try dout2.matmul(w_col_t);
            defer dcol.deinit();

            const dx = try tools.col2im(
                N,
                C,
                H,
                W,
                FH,
                FW,
                pads,
                stride,
                T,
                self.allocator,
                dcol,
                self.shape_env,
            );

            return dx;
        }
    };
}

test "Convolution" {
    const allocator = std.testing.allocator;

    {
        const N = comptime SizeExpr.sym(.{ .name = "batch_size" });
        const C = comptime SizeExpr.static(2);
        const H = comptime SizeExpr.static(4);
        const W = comptime SizeExpr.static(4);

        std.debug.print("N: {f}\n", .{N});

        const FN = comptime SizeExpr.sym(.{ .name = "filter_num" });
        const FH = comptime SizeExpr.static(3);
        const FW = comptime SizeExpr.static(3);

        const PAD = comptime SizeExpr.sym(.{ .name = "pad" });
        // const PAD = comptime SizeExpr.static(1);
        // const STRIDE = comptime SizeExpr.static(1);
        const STRIDE = comptime SizeExpr.sym(.{ .name = "stride" });

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
        std.debug.print("OH= {f} OW= {f}\n", .{ Conv.F_OH, Conv.F_OW });

        var shape_env = try shape_expr.ShapeEnv.init(allocator);
        defer shape_env.deinit();

        try shape_env.bind(&N.Sym, 1);
        try shape_env.bind(&FN.Sym, 1);
        try shape_env.bind(&PAD.Sym, 1);
        try shape_env.bind(&STRIDE.Sym, 1);

        var w = try tensor.fromArray(allocator, [2][4][4]f32{
            [4][4]f32{
                [4]f32{ 1.0, 2.0, 3.0, 0.0 },
                [4]f32{ 0.0, 1.0, 2.0, 3.0 },
                [4]f32{ 3.0, 0.0, 1.0, 2.0 },
                [4]f32{ 2.0, 3.0, 0.0, 1.0 },
            },
            [4][4]f32{
                [4]f32{ 2.0, 4.0, 6.0, 0.0 },
                [4]f32{ 0.0, 1.0, 2.0, 3.0 },
                [4]f32{ 3.0, 0.0, 1.0, 2.0 },
                [4]f32{ 2.0, 3.0, 0.0, 1.0 },
            },
        }, &shape_env);
        defer w.deinit();

        var w_f = try w.reshape(&.{ N, C, H, W });
        defer w_f.deinit();

        var filter = try tensor.fromArray(allocator, [2][3][3]f32{
            [3][3]f32{
                [3]f32{ 2.0, 0.0, 1.0 },
                [3]f32{ 0.0, 1.0, 2.0 },
                [3]f32{ 1.0, 0.0, 2.0 },
            },
            [3][3]f32{
                [3]f32{ 2.0, 0.0, 1.0 },
                [3]f32{ 0.0, 2.0, 4.0 },
                [3]f32{ 1.0, 0.0, 2.0 },
            },
        }, &shape_env);
        defer filter.deinit();

        const filter_f = try filter.reshape(&.{ FN, C, FH, FW });

        const b = try tensor.full(allocator, &.{FN}, &shape_env, @as(f32, 3.0));

        var conv_layer = Conv.initImpl(allocator, &shape_env, filter_f, b);
        defer conv_layer.deinit();

        var result = try conv_layer.forward(&w_f);
        defer result.deinit();

        std.debug.print("result: {f}\n", .{result});

        const result_view = result.view();

        const expected = try tensor.fromArray(allocator, [1][1][4][4]f32{
            [1][4][4]f32{[4][4]f32{
                [4]f32{ 32, 51, 32, 7 },
                [4]f32{ 15, 43, 47, 32 },
                [4]f32{ 26, 17, 38, 17 },
                [4]f32{ 27, 26, 13, 10 },
            }},
        }, &shape_env);
        defer expected.deinit();
        const expected_view = expected.view();

        const equal_res = expected_view.equal(&result_view);
        try std.testing.expect(equal_res);
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

pub fn Pooling(
    comptime N: SizeExpr,
    comptime C: SizeExpr,
    comptime H: SizeExpr,
    comptime W: SizeExpr,
    comptime FH: SizeExpr,
    comptime FW: SizeExpr,
    comptime pads: [4]SizeExpr,
    comptime stride: SizeExpr,
    comptime T: type,
) type {
    return struct {
        const F_OH = H.add(pads[2]).add(pads[3]).sub(FH).div(stride).add(SizeExpr.static(1));
        const F_OW = W.add(pads[0]).add(pads[1]).sub(FW).div(stride).add(SizeExpr.static(1));

        const Self = @This();

        allocator: std.mem.Allocator,
        shape_env: *const shape_expr.ShapeEnv,
        arg_max: ?tensor.Tensor(&.{ N.mul(F_OH).mul(F_OW).mul(C), SizeExpr.static(1) }, [2]usize) = null,

        pub fn deinit(self: *Self) void {
            if (self.arg_max) |arg_max| {
                arg_max.deinit();
            }
        }

        pub fn init(allocator: std.mem.Allocator, shape_env: *const shape_expr.ShapeEnv) Self {
            return Self{
                .allocator = allocator,
                .shape_env = shape_env,
            };
        }

        pub fn forward(self: *Self, x: *const tensor.Tensor(&.{ N, C, H, W }, T)) !tensor.Tensor(&.{ N, C, F_OH, F_OW }, T) {
            const col_raw = try tools.im2col(
                N,
                C,
                H,
                W,
                FH,
                FW,
                pads,
                stride,
                T,
                self.allocator,
                x,
                self.shape_env,
            );
            defer col_raw.deinit();

            std.debug.print("col raw: {f}\n", .{col_raw});

            const col_data = try col_raw.reshape(&.{ N.mul(F_OH).mul(F_OW).mul(C), FH.mul(FW) });
            defer col_data.deinit();

            std.debug.print("col data: {f}\n", .{col_data});

            const out1 = try col_data.max(1);
            defer out1.deinit();

            const arg_out = try col_data.argMax(1);

            // comptime {
            //     @setEvalBranchQuota(20000);
            //     @compileLog("shape info: " ++ "self_arg_max= " ++ shape_expr.compLog(@TypeOf(self.arg_max.?).S));
            //     @compileLog("shape info: " ++ "arg_out= " ++ shape_expr.compLog(@TypeOf(arg_out).S));
            // }

            self.arg_max = arg_out;

            const out2 = try out1.reshape(&.{ N, F_OH, F_OW, C });
            defer out2.deinit();
            const out3 = out2.permute([4]usize{ 0, 3, 1, 2 });

            return out3;
        }

        pub fn backward(self: *Self, dout: *const tensor.Tensor(&.{ N, C, F_OH, F_OW }, T)) !tensor.Tensor(&.{ N, C, H, W }, T) {
            const dout1 = dout.permute([4]usize{ 0, 2, 3, 1 });
            defer dout1.deinit();

            var dmax1 = try tensor.zeros(self.allocator, T, &.{ @TypeOf(dout1).sizeExpr(), FH.mul(FW) }, self.shape_env);
            defer dmax1.deinit();

            for (self.arg_max.?.dataSliceRaw(), dout1.dataSliceRaw()) |arg_max, dout_v| {
                try dmax1.setData(&arg_max, dout_v);
            }

            var dmax2 = try dmax1.reshape(&.{ N, F_OH, F_OW, C, FH, FW });
            defer dmax2.deinit();

            var dcol = try dmax2.reshape(&.{ N.mul(F_OH).mul(F_OW), C.mul(FH).mul(FW) });
            defer dcol.deinit();

            const dx = try tools.col2im(
                N,
                C,
                H,
                W,
                FH,
                FW,
                pads,
                stride,
                T,
                self.allocator,
                &dcol,
                self.shape_env,
            );

            return dx;
        }
    };
}

test "Pooling" {
    const allocator = std.testing.allocator;

    const N = comptime SizeExpr.sym(.{ .name = "batch_size" });
    const C = comptime SizeExpr.static(3);
    const H = comptime SizeExpr.static(4);
    const W = comptime SizeExpr.static(4);

    std.debug.print("N: {f}\n", .{N});

    const FN = comptime SizeExpr.sym(.{ .name = "filter_num" });
    const FH = comptime SizeExpr.static(2);
    const FW = comptime SizeExpr.static(2);

    const PAD = comptime SizeExpr.sym(.{ .name = "pad" });
    // const PAD = comptime SizeExpr.static(1);
    // const STRIDE = comptime SizeExpr.static(1);
    const STRIDE = comptime SizeExpr.sym(.{ .name = "stride" });

    std.debug.print(
        "N= {f} C= {f} H= {f} W= {f} FN= {f} FH= {f} FW= {f}\n",
        .{ N, C, H, W, FN, FH, FW },
    );

    const Pool = Pooling(
        N,
        C,
        H,
        W,
        FH,
        FW,
        [4]SizeExpr{ PAD, PAD, PAD, PAD },
        STRIDE,
        f32,
    );

    var shape_env = try shape_expr.ShapeEnv.init(allocator);
    defer shape_env.deinit();

    try shape_env.bind(&N.Sym, 1);
    try shape_env.bind(&FN.Sym, 1);
    try shape_env.bind(&PAD.Sym, 0);
    try shape_env.bind(&STRIDE.Sym, 2);

    var w = try tensor.fromArray(allocator, [3][4][4]f32{
        [4][4]f32{
            [4]f32{ 1.0, 2.0, 4.0, 2.0 },
            [4]f32{ 0.0, 1.0, 0.0, 1.0 },
            [4]f32{ 3.0, 0.0, 1.0, 2.0 },
            [4]f32{ 2.0, 3.0, 0.0, 4.0 },
        },
        [4][4]f32{
            [4]f32{ 3.0, 0.0, 3.0, 0.0 },
            [4]f32{ 2.0, 4.0, 4.0, 2.0 },
            [4]f32{ 1.0, 0.0, 3.0, 0.0 },
            [4]f32{ 3.0, 1.0, 4.0, 2.0 },
        },
        [4][4]f32{
            [4]f32{ 1.0, 0.0, 6.0, 5.0 },
            [4]f32{ 3.0, 2.0, 4.0, 3.0 },
            [4]f32{ 4.0, 2.0, 6.0, 2.0 },
            [4]f32{ 0.0, 1.0, 4.0, 5.0 },
        },
    }, &shape_env);
    defer w.deinit();

    var x = try w.reshape(&.{ N, C, H, W });
    defer x.deinit();

    var pool_layer = Pool.init(allocator, &shape_env);
    defer pool_layer.deinit();

    std.debug.print("input: {f}\n", .{x});
    var result = try pool_layer.forward(&x);
    defer result.deinit();

    std.debug.print("result: {f}\n", .{result});

    const result_view = result.view();

    const expected = try tensor.fromArray(allocator, [1][1][4][4]f32{
        [1][4][4]f32{[4][4]f32{
            [4]f32{ 32, 51, 32, 7 },
            [4]f32{ 15, 43, 47, 32 },
            [4]f32{ 26, 17, 38, 17 },
            [4]f32{ 27, 26, 13, 10 },
        }},
    }, &shape_env);
    defer expected.deinit();
    const expected_view = expected.view();

    const equal_res = expected_view.equal(&result_view);
    try std.testing.expect(equal_res);
}
