const std = @import("std");
const dvui = @import("dvui");
const SDLBackend = @import("sdl-backend");

var win: dvui.Window = undefined;

var allocator: std.mem.Allocator = undefined;

const Pair = struct {
    x: std.ArrayList(f64),
    y: std.ArrayList(f64),

    pub fn init(allocator_a: std.mem.Allocator) !Pair {
        return Pair{
            .x = try std.ArrayList(f64).initCapacity(allocator_a, 10),
            .y = try std.ArrayList(f64).initCapacity(allocator_a, 10),
        };
    }

    pub fn axisInfo(self: *const @This()) struct { x_min: f64, x_max: f64, y_min: f64, y_max: f64 } {
        const x_min_i = minInSlice(self.x.items);
        const x_max_i = maxInSlice(self.x.items);
        const y_min_i = minInSlice(self.y.items);
        const y_max_i = maxInSlice(self.y.items);

        return .{ .x_min = x_min_i - @abs(x_min_i) * 0.1, .x_max = x_max_i + @abs(x_max_i) * 0.1, .y_min = y_min_i - @abs(y_min_i) * 0.1, .y_max = y_max_i + @abs(y_max_i) * 0.1 };
    }
};

var datasets: std.StringHashMap(Pair) = undefined;
var rwlock: std.Thread.RwLock = std.Thread.RwLock{};

pub fn appendData(key: []const u8, xval_s: []const f64, yval_s: []const f64) !void {
    rwlock.lock();
    defer rwlock.unlock();

    var entry = try datasets.getOrPutValue(key, try Pair.init(allocator));
    try entry.value_ptr.x.appendSlice(allocator, xval_s);
    try entry.value_ptr.y.appendSlice(allocator, yval_s);

    dvui.refresh(&win, @src(), null);
}

pub fn beginPlotLoop(allocator_a: std.mem.Allocator) !void {
    allocator = allocator_a;

    datasets = std.StringHashMap(Pair).init(allocator);

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

        try plotImpl();

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

fn plotImpl() !void {
    rwlock.lockShared();
    const datasets_i = try datasets.clone();
    rwlock.unlockShared();

    var data_iter = datasets_i.iterator();

    var axis_info = if (data_iter.next()) |entry| entry.value_ptr.axisInfo() else return;

    while (data_iter.next()) |entry| {
        const axis_info_n = entry.value_ptr.axisInfo();

        if (axis_info_n.x_min < axis_info.x_min) axis_info.x_min = axis_info_n.x_min;
        if (axis_info_n.x_max > axis_info.x_max) axis_info.x_max = axis_info_n.x_max;
        if (axis_info_n.y_min < axis_info.y_min) axis_info.y_min = axis_info_n.y_min;
        if (axis_info_n.y_max > axis_info.y_max) axis_info.y_max = axis_info_n.y_max;
    }

    const gridline_color = dvui.Color.fromHSLuv(0, 0, 0, 100);
    const subtick_gridline_color = dvui.Color.fromHSLuv(0, 0, 0, 30);

    var x_axis: dvui.PlotWidget.Axis = .{ .name = "X Axis", .min = axis_info.x_min, .max = axis_info.x_max, .ticks = .{
        .side = .left_or_top,
        .subticks = true,
    }, .gridline_color = gridline_color, .subtick_gridline_color = subtick_gridline_color };
    var y_axis: dvui.PlotWidget.Axis = .{ .name = "Y Axis", .min = axis_info.y_min, .max = axis_info.y_max, .ticks = .{ .side = .both, .subticks = true }, .gridline_color = gridline_color, .subtick_gridline_color = subtick_gridline_color };

    var data_iter_n = datasets_i.iterator();

    var plot = dvui.plot(@src(), .{
        .x_axis = &x_axis,
        .y_axis = &y_axis,
        .border_thick = 1.0,
    }, .{ .expand = .both });
    defer plot.deinit();

    while (data_iter_n.next()) |entry| {
        var s = plot.line();
        defer s.deinit();

        for (entry.value_ptr.x.items, entry.value_ptr.y.items) |x, y| {
            s.point(x, y);
        }

        s.stroke(1.3, autoColor());
    }

    next_auto_color_idx = 0;
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
