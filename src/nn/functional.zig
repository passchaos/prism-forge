const std = @import("std");

const Tensor = @import("../Tensor.zig");
const dtype = @import("../dtype.zig");
const DataType = dtype.DataType;
const utils = @import("../utils.zig");
const ShapeIterator = @import("../Layout.zig").ShapeIterator;

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
