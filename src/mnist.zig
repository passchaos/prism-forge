const std = @import("std");
const tensor = @import("tensor.zig");
const layout = @import("layout.zig");
const storage = @import("storage.zig");
const log = @import("log.zig");

const Tensor_2 = tensor.Tensor(2, .{});
const Tensor_1 = tensor.Tensor(1, .{});

pub fn loadImages(allocator: std.mem.Allocator, path: []const u8, comptime shape: []const usize) !tensor.Tensor(shape, u8, .{}) {
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

    const read_shape = &.{ @as(usize, @intCast(samples)), @as(usize, @intCast(rows * columns)) };

    if (!std.mem.eql(usize, read_shape, shape)) {
        return error.ShapeMismatch;
    }

    return try tensor.fromData(u8, allocator, data, shape);
}

pub fn loadLabels(allocator: std.mem.Allocator, path: []const u8, comptime count: usize) !tensor.Tensor(&.{count}, u8, .{}) {
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

    if (samples != count) {
        return error.ShapeMismatch;
    }

    return try tensor.fromData(u8, allocator, new_buf, &.{count});
}

pub fn loadDatas(comptime T: type, allocator: std.mem.Allocator) !struct {
    train_images: tensor.Tensor(&.{ 60000, 784 }, T, .{}),
    train_labels: tensor.Tensor(&.{ 60000, 10 }, T, .{}),
    test_images: tensor.Tensor(&.{ 10000, 784 }, T, .{}),
    test_labels: tensor.Tensor(&.{ 10000, 10 }, T, .{}),
} {
    const env_home = std.posix.getenv("HOME").?;

    const path_images = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/train-images.idx3-ubyte" });
    defer allocator.free(path_images);

    var train_images = try loadImages(allocator, path_images, &.{ 60000, 784 });
    defer train_images.deinit();

    const func = struct {
        fn call(v: u8, _: void) T {
            return @as(T, @floatFromInt(v)) / 255.0;
        }
    }.call;
    const train_images_one = try train_images.map(void{}, T, func);

    const path_labels = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/train-labels.idx1-ubyte" });
    defer allocator.free(path_labels);

    const train_labels = try loadLabels(allocator, path_labels, 60000);
    defer train_labels.deinit();
    const train_labels_oh = try train_labels.oneHot(T, 10);

    const path_test_images = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/t10k-images.idx3-ubyte" });
    defer allocator.free(path_test_images);

    var test_images = try loadImages(allocator, path_test_images, &.{ 10000, 784 });
    defer test_images.deinit();

    const test_images_one = try test_images.map(void{}, T, func);

    const path_test_labels = try std.fs.path.join(allocator, &.{ env_home, "Work/mnist/t10k-labels.idx1-ubyte" });
    defer allocator.free(path_test_labels);

    const test_labels = try loadLabels(allocator, path_test_labels, 10000);
    defer test_labels.deinit();
    const test_labels_oh = try test_labels.oneHot(T, 10);

    return .{
        .train_images = train_images_one,
        .train_labels = train_labels_oh,
        .test_images = test_images_one,
        .test_labels = test_labels_oh,
    };
}

test "mnist images and labels" {
    const allocator = std.testing.allocator;

    const res = try loadDatas(allocator);

    const train_images = res.train_images;
    defer train_images.deinit();
    const train_labels = res.train_labels;
    defer train_labels.deinit();
    const test_images = res.test_images;
    defer test_images.deinit();
    const test_labels = res.test_labels;
    defer test_labels.deinit();

    log.print(@src(), "train_images: {f} train_labels: {f} test_images: {f} test_labels: {f}\n", .{ train_images, train_labels, test_images, test_labels });
}
