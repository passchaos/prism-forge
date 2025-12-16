const std = @import("std");
const dvui = @import("dvui");
const SDLBackend = @import("sdl-backend");

var win: dvui.Window = undefined;

var allocator: std.mem.Allocator = undefined;

var xvals: std.ArrayList(f64) = undefined;
var yvals: std.ArrayList(f64) = undefined;

pub fn appendData(xval_s: []const f64, yval_s: []const f64) !void {
    try xvals.appendSlice(allocator, xval_s);
    try yvals.appendSlice(allocator, yval_s);

    dvui.refresh(&win, @src(), null);
}

pub fn beginPlotLoop(allocator_a: std.mem.Allocator) !void {
    allocator = allocator_a;

    xvals = try std.ArrayList(f64).initCapacity(allocator, 10);
    yvals = try std.ArrayList(f64).initCapacity(allocator, 10);

    var backend = try SDLBackend.initWindow(.{ .allocator = allocator, .size = .{ .w = 800.0, .h = 600.0 }, .vsync = true, .title = "prism plot" });
    defer backend.deinit();

    win = try dvui.Window.init(@src(), allocator, backend.backend(), .{ .theme = dvui.Theme.builtin.adwaita_light });
    defer win.deinit();

    var interrupted = false;

    main_loop: while (true) {
        const nstime = win.beginWait(interrupted);

        try win.begin(nstime);
        try backend.addAllEvents(&win);

        _ = SDLBackend.c.SDL_SetRenderDrawColor(backend.renderer, 255, 255, 255, 255);
        _ = SDLBackend.c.SDL_RenderClear(backend.renderer);

        plotImpl(xvals.items, yvals.items);

        for (dvui.events()) |event| {
            switch (event.evt) {
                .window => if (event.evt.window.action == .close) {
                    std.debug.print("window close\n", .{});
                    break :main_loop;
                },
                .app => if (event.evt.app.action == .quit) {
                    std.debug.print("app quit\n", .{});
                    break :main_loop;
                },
                else => {},
            }
        }

        const end_micros = try win.end(.{});

        try backend.setCursor(win.cursorRequested());
        try backend.textInputRect(win.textInputRequested());
        try backend.renderPresent();

        const wait_event_micros = win.waitTime(end_micros);
        interrupted = try backend.waitEventTimeout(wait_event_micros);
    }
}

fn plotImpl(xval_a: []f64, yval_a: []f64) void {
    var x_min = minInSlice(xval_a);
    x_min = x_min - @abs(x_min) * 0.1;

    var x_max = maxInSlice(xval_a);
    x_max = x_max + @abs(x_max) * 0.1;

    var y_min = minInSlice(yval_a);
    y_min = y_min - @abs(y_min) * 0.1;

    var y_max = maxInSlice(yval_a);
    y_max = y_max + @abs(y_max) * 0.1;

    const gridline_color = dvui.Color.fromHSLuv(0, 0, 0, 100);
    const subtick_gridline_color = dvui.Color.fromHSLuv(0, 0, 0, 30);

    var x_axis: dvui.PlotWidget.Axis = .{ .name = "X Axis", .min = x_min, .max = x_max, .ticks = .{
        .side = .left_or_top,
        .subticks = true,
    }, .gridline_color = gridline_color, .subtick_gridline_color = subtick_gridline_color };
    var y_axis: dvui.PlotWidget.Axis = .{ .name = "Y Axis", .min = y_min, .max = y_max, .ticks = .{ .side = .both, .subticks = true }, .gridline_color = gridline_color, .subtick_gridline_color = subtick_gridline_color };

    dvui.plotXY(@src(), .{
        .plot_opts = .{ .x_axis = &x_axis, .y_axis = &y_axis, .border_thick = 1.0, .spine_color = subtick_gridline_color },
        .xs = xval_a,
        .ys = yval_a,
        .thick = 1.3,
    }, .{ .expand = .both, .gravity_x = 0.5, .gravity_y = 0.5 });
}

fn minInSlice(slice: []const f64) f64 {
    if (slice.len == 0) return 0.0;

    var min = slice[0];
    for (slice[1..]) |val| {
        if (val < min) min = val;
    }
    return min;
}

fn maxInSlice(slice: []const f64) f64 {
    if (slice.len == 0) return 0.0;

    var max = slice[0];
    for (slice[1..]) |val| {
        if (val > max) max = val;
    }
    return max;
}

var next_auto_color_idx: usize = 0;
fn autoColor() dvui.Color {
    const i = next_auto_color_idx;
    next_auto_color_idx += 1;

    const golden_ratio = comptime (std.math.sqrt(5.0) - 1.0);
    const hue = @mod(@as(f32, @floatFromInt(i)) * golden_ratio * 360.0, 360.0);

    const hsv_color = dvui.Color.HSV{
        .h = hue,
        .s = 0.85,
        .v = 0.5,
        .a = 1.0,
    };
    return hsv_color.toColor();
}
