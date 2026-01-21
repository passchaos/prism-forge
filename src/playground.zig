const std = @import("std");
const shape_expr = @import("shape_expr.zig");

const SizeExpr = shape_expr.SizeExpr;

const Layer = enum {
    Relu,
    Affine,
};

fn Relu(comptime shape_expr_a: []const SizeExpr, comptime T: type) type {
    return struct {
        const S = shape_expr_a;
        tag: Layer,
        v: T,

        const Self = @This();
        pub fn forward(self: *Self, x: *const shape_expr_a[0], t: *const shape_expr_a[1]) *const shape_expr_a[1] {
            _ = self;
            _ = x;
            // _ = t;
            return t;
        }

        pub fn backward(self: *Self, dout: *const shape_expr_a[0]) *const shape_expr_a[0] {
            _ = self;
            // _ = dout;
            return dout;
        }
    };
}

fn Affine(comptime shape_expr_a: []const SizeExpr, comptime T: type) type {
    return struct {
        const S = shape_expr_a;
        tag: Layer,
        v: T,

        const Self = @This();
        pub fn forward(self: *Self, x: *const shape_expr_a[0], t: *const shape_expr_a[1]) *const shape_expr_a[1] {
            _ = self;
            _ = x;
            return t;
        }

        pub fn backward(self: *Self, dout: *const shape_expr_a[0]) *const shape_expr_a[0] {
            _ = self;
            return dout;
        }
    };
}

test "field_parent" {
    // const allocator = std.testing.allocator;

    const ReluL = Relu(&.{ SizeExpr.static(2), SizeExpr.static(1) }, f32);
    const AffineL = Affine(&.{ SizeExpr.static(3), SizeExpr.static(3) }, f32);

    var relu = ReluL{ .tag = .Relu, .v = 10.0 };
    var affine = AffineL{ .tag = .Affine, .v = 20.0 };

    const layers = [2]*Layer{ &relu.tag, &affine.tag };

    for (layers) |layer| {
        switch (layer.*) {
            .Relu => {
                const parent: *ReluL = @alignCast(@fieldParentPtr("tag", layer));
                std.debug.print("Relu Layer: {} {any}\n", .{ parent, @TypeOf(parent.*).S });
            },
            .Affine => {
                const parent: *AffineL = @alignCast(@fieldParentPtr("tag", layer));
                std.debug.print("Affine Layer: {}\n", .{parent});
            },
        }
    }
}
