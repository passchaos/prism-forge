const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const prism_mod = b.addModule("prism", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // setup blas linkage
    switch (target.result.os.tag) {
        .linux => {
            prism_mod.link_libc = true;
            prism_mod.linkSystemLibrary("openblas", .{});
        },
        .macos => {
            prism_mod.linkFramework("Accelerate", .{});
        },
        else => {},
    }

    const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .sdl3 });

    prism_mod.addImport("dvui", dvui_dep.module("dvui_sdl3"));
    prism_mod.addImport("sdl-backend", dvui_dep.module("sdl3"));

    const zeit_dep = b.dependency("zeit", .{ .target = target, .optimize = optimize });
    const zeit_mod = zeit_dep.module("zeit");
    prism_mod.addImport("zeit", zeit_mod);

    const zdt_dep = b.dependency("zdt", .{ .target = target, .optimize = optimize });
    const zdt_mod = zdt_dep.module("zdt");
    prism_mod.addImport("zdt", zdt_mod);

    const mnist_exe = b.addExecutable(.{
        .name = "mnist",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/mnist.zig"),
            .imports = &.{.{ .name = "prism", .module = prism_mod }},
            .target = target,
        }),
    });
    const mnist_cmd = b.addRunArtifact(mnist_exe);
    const mnist_step = b.step("mnist", "Run mnist");
    mnist_step.dependOn(&mnist_cmd.step);

    const exe_check = b.addExecutable(.{ .name = "exe_check", .root_module = prism_mod });
    const check = b.step("check", "Check if exe_check compiles");
    check.dependOn(&exe_check.step);

    const mod_tests = b.addTest(.{
        .root_module = prism_mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_filters = b.option(
        []const []const u8,
        "test-filter",
        "Filter test cases",
    ) orelse &[0][]const u8{};
    const exe_tests = b.addTest(.{
        .filters = test_filters,
        .root_module = prism_mod,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
