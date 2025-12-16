const std = @import("std");
const Tensor = @import("../Tensor.zig");
const DataType = @import("../dtype.zig").DataType;

pub fn WeightInit(comptime T: type) type {
    return union(enum) { std: T, xavier, he };
}

pub const Linear = struct {
    allocator: std.mem.Allocator,
    _weight: Tensor,
    bias: ?Tensor,
    x: ?Tensor,
    x_original_shape: ?[]const usize,

    const Self = @This();

    pub fn new(allocator: std.mem.Allocator, comptime T: type, weight_init: WeightInit(T), input_size: usize, output_size: usize) !Self {
        switch (@typeInfo(T)) {
            inline .float => {
                var weights = try Tensor.randNorm(allocator, T, &.{ input_size, output_size }, 0.0, 1.0);

                const scale = switch (weight_init) {
                    .std => |std_v| std_v,
                    .xavier => 6.0 / @as(T, @floatFromInt(input_size + output_size)),
                    .he => 2.0 / @sqrt(input_size),
                };

                try weights.mul_(scale);

                const bias = try Tensor.zeros(allocator, T, &.{output_size});

                return Self{
                    .allocator = allocator,
                    ._weight = weights,
                    .bias = bias,
                    .x = null,
                    .x_original_shape = null,
                };
            },
            inline else => @compileError("Unsupported type" ++ @typeName(T)),
        }
    }

    pub fn forward(self: *Self, input: Tensor) !Tensor {
        self.x_original_shape = try self.allocator.dupe(usize, input.shapes());

        self.x = try input.clone();

        const out = if (self.bias) |bias|
            try (try input.matmul(&self._weight)).add(bias)
        else
            try input.matmul(&self._weight);
        return out;
    }

    pub fn backward(self: *Self, grad: Tensor) !Tensor {
        var wi = try self._weight.clone();
        try wi.transpose_();

        const dx = try grad.matmul(&wi);

        var x_t = try self.x.?.clone();
        try x_t.transpose_();

        // const dw = try x_t.matmul(&grad);

        return dx;
    }
};
