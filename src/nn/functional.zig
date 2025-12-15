const std = @import("std");

const Tensor = @import("../Tensor.zig");
const dtype = @import("../dtype.zig");
const DataType = dtype.DataType;
const utils = @import("../utils.zig");
const ShapeIterator = @import("../Layout.zig").ShapeIterator;

pub fn oneHot(self: Tensor, args: struct { num_classes: ?usize = null }) !Tensor {
    switch (self.dtype()) {
        inline .u8, .i32, .u32 => |v| {
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
