const std = @import("std");
const utils = @import("utils.zig");
const host = @import("./device/host.zig");
const log = @import("./log.zig");

const dtype_o = @import("./dtype.zig");
const DataType = dtype_o.DataType;
const Scalar = dtype_o.Scalar;

const storage_t = @import("./storage.zig");
const Device = storage_t.Device;
const layout_t = @import("./layout.zig");

const shape_expr = @import("shape_expr.zig");
const SizeExpr = shape_expr.SizeExpr;
const ShapeEnv = shape_expr.ShapeEnv;
const parseSpec = shape_expr.parseSpec;

// 核心设计：
// Tensor: 完全动态张量
// 