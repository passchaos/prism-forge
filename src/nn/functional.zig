const std = @import("std");

const Tensor = @import("../Tensor.zig");
const DataType = @import("../dtype.zig").DataType;

pub fn oneHot(self: Tensor, args: struct { num_classes: ?usize = null }) !Tensor {
    switch (self.dtype()) {
        inline .i32, .u32 => |v| {
            const T = v.toTypeComp();
            const nc: usize = if (args.num_classes) |nc| nc else blk: {
                const max_t = try self.max(null);
                const item = try max_t.scalarItemComp(v) + 1;

                break :blk @intCast(item);
            };

            var result_shapes = try std.ArrayList(usize).initCapacity(self.allocator, 10);
            try result_shapes.appendSlice(self.allocator, self.shapes());
            try result_shapes.append(self.allocator, nc);

            var result = try Tensor.full(self.allocator, result_shapes.items, @as(T, 0));

            var self_iter = try self.dataIter();
            defer self_iter.deinit();

            const dims = result.ndim();
            var indices = try self.allocator.alloc(usize, dims);
            while (self_iter.next()) |idx| {
                @memcpy(indices[0 .. dims - 1], idx);
                indices[dims - 1] = @intCast((try self.getWithIndicesCompType(v, idx)).*);

                (try result.getWithIndicesCompType(v, indices)).* = @intCast(1);
            }

            return result;
        },
        else => return error.InvalidDataType,
    }
}
