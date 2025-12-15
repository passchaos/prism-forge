const std = @import("std");
const Tensor = @import("Tensor.zig");
const Layout = @import("Layout.zig");
const Storage = @import("Storage.zig");
const DataType = @import("dtype.zig").DataType;
const Scalar = @import("dtype.zig").Scalar;
const F = @import("nn/functional.zig");

pub fn loadImages(allocator: std.mem.Allocator, path: []const u8) !Tensor {
    var buf = [_]u8{0} ** 4;

    const file = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });

    const nm = try file.read(buf[0..]);
    std.debug.assert(nm == 4);

    const magic_number = std.mem.readInt(u32, &buf, .big);
    if (magic_number != 2051) {
        return error.NotImages;
    }

    const ns = try file.read(buf[0..]);
    std.debug.assert(ns == 4);
    const samples = std.mem.readInt(u32, &buf, .big);

    const nr = try file.read(buf[0..]);
    std.debug.assert(nr == 4);
    const rows = std.mem.readInt(u32, &buf, .big);

    const nc = try file.read(buf[0..]);
    std.debug.assert(nc == 4);
    const columns = std.mem.readInt(u32, &buf, .big);

    const data_size = samples * rows * columns;
    const data = try file.readToEndAlloc(allocator, data_size);

    const layout = try Layout.init(allocator, DataType.u8, &.{ samples, rows * columns });
    const storage = Storage.init(allocator, Storage.Device.Cpu, data.ptr, data.len);

    return try Tensor.fromDataImpl(allocator, layout, storage, 0);
}

pub fn loadLabels(allocator: std.mem.Allocator, path: []const u8) !Tensor {
    var buf = [_]u8{0} ** 4;

    const file = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });

    const nm = try file.read(buf[0..]);
    std.debug.assert(nm == 4);

    const magic_number = std.mem.readInt(u32, &buf, .big);
    if (magic_number != 2049) {
        return error.NotImages;
    }

    _ = try file.read(buf[0..4]);
    const samples = std.mem.readInt(u32, &buf, .big);

    const new_buf = try file.readToEndAlloc(allocator, 100 * 1024 * 1024);
    std.debug.assert(new_buf.len == samples);

    const layout = try Layout.init(allocator, DataType.u8, &.{new_buf.len});
    const storage = Storage.init(allocator, Storage.Device.Cpu, new_buf.ptr, new_buf.len);

    return try Tensor.fromDataImpl(allocator, layout, storage, 0);
}

pub fn loadDatas(allocator: std.mem.Allocator) !struct {
    train_images: Tensor,
    train_labels: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
} {
    const env_home = std.posix.getenv("HOME").?;

    const path_images = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/train-images.idx3-ubyte" });
    var train_images = try loadImages(allocator, path_images);

    const func = struct {
        fn call(v: u8) f32 {
            return @as(f32, @floatFromInt(v)) / 255.0;
        }
    }.call;
    const train_images_one = try train_images.map(DataType.u8, DataType.f32, func);

    const path_labels = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/train-labels.idx1-ubyte" });
    const train_labels = try loadLabels(allocator, path_labels);
    const train_labels_oh = try F.oneHot(train_labels, .{ .num_classes = 10 });

    const path_test_images = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/t10k-images.idx3-ubyte" });
    var test_images = try loadImages(allocator, path_test_images);
    const test_images_one = try test_images.map(DataType.u8, DataType.f32, func);

    const path_test_labels = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/t10k-labels.idx1-ubyte" });
    const test_labels = try loadLabels(allocator, path_test_labels);
    const test_labels_oh = try F.oneHot(test_labels, .{ .num_classes = 10 });

    return .{
        .train_images = train_images_one,
        .train_labels = train_labels_oh,
        .test_images = test_images_one,
        .test_labels = test_labels_oh,
    };
}

test "mnist images and labels" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();
    const res = try loadDatas(allocator);

    const train_images = res.train_images;
    const train_labels = res.train_labels;
    const test_images = res.test_images;
    const test_labels = res.test_labels;

    std.debug.print("train_images: {f} train_labels: {f} test_images: {f} test_labels: {f}\n", .{ train_images, train_labels, test_images, test_labels });
}
