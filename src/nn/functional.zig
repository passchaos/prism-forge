const std = @import("std");

const Tensor = @import("../Tensor.zig");
const dtype = @import("../dtype.zig");
const DataType = dtype.DataType;
const utils = @import("../utils.zig");
const ShapeIterator = @import("../Layout.zig").ShapeIterator;

pub fn pad(self: Tensor, pads: []const usize, value: anytype) !Tensor {
    switch (self.dtype()) {
        inline else => |dt| {
            const T = dt.toTypeComp();

            const vv = comptime dtype.toDType(T, value);

            const dims = self.ndim();
            const pad_dims = pads.len / 2;

            var new_shapes = try std.ArrayList(usize).initCapacity(self.allocator, dims);
            try new_shapes.appendSlice(self.allocator, self.shapes());

            for (new_shapes.items, 0..) |*shape, i| {
                const idx_to_pad_idx = dims - i - 1;

                if (idx_to_pad_idx < pad_dims) {
                    const left_add = pads[2 * idx_to_pad_idx];
                    const right_add = pads[2 * idx_to_pad_idx + 1];
                    shape.* += left_add + right_add;
                }
            }

            var result = try Tensor.full(self.allocator, new_shapes.items, vv);

            var shape_iter = try ShapeIterator.init(self.allocator, self.shapes());
            while (shape_iter.next()) |idx| {
                var dst_idx = try self.allocator.alloc(usize, dims);

                for (idx, 0..) |dim, i| {
                    // judge if need to set value from orig tensor
                    const idx_to_pad_idx = dims - i - 1;

                    if (idx_to_pad_idx < pad_dims) {
                        const left_add = pads[2 * idx_to_pad_idx];
                        dst_idx[i] = dim + left_add;
                    } else {
                        dst_idx[i] = dim;
                    }
                }

                const dst_v = try result.getWithIndicesCompType(dt, dst_idx);
                const src_v = try self.getWithIndicesCompType(dt, idx);

                dst_v.* = src_v.*;
            }

            return result;
        },
    }
}

// loss
pub fn mseLoss(self: Tensor, other: Tensor) !Tensor {
    var a = try self.sub(other);
    try a.powi_(2);

    return try a.sum(null);
}

pub fn crossEntropy(self: Tensor, other: Tensor) !Tensor {
    switch (self.dtype()) {
        inline .f32 => |dt| {
            const scope = struct {
                fn call(v: dt.toTypeComp(), _: void) dt.toTypeComp() {
                    return -@log(v + 0.0001);
                }
            };

            const a = try self.map(dt, dt, void{}, scope.call);

            const a1 = try a.mul(other);
            var value = try a1.sum(null);

            const batch_size = switch (self.ndim()) {
                1 => 1,
                2 => self.shapes()[0],
                else => return error.InvalidDimension,
            };

            try value.div_(batch_size);
            return value;
        },
        inline else => return error.InvalidDataType,
    }
}
