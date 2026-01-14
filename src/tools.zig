const std = @import("std");

pub const IDGenerator = struct {
    // 编译时计数器（仅编译时可见，无原子操作）
    comptime_counter: u64 = 0,
    // 运行时原子计数器（仅运行时生效）
    runtime_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    /// 通用next方法：自动适配编译时/运行时
    pub fn next(self: *@This()) u64 {
        if (@inComptime()) {
            // 编译时：直接自增编译时计数器（无并发，无需原子操作）
            const id = self.comptime_counter;
            self.comptime_counter += 1;
            return id;
        } else {
            // 运行时：使用原子操作自增（线程安全）
            const id = self.runtime_counter.fetchAdd(1, .SeqCst);
            return id;
        }
    }

    /// 重置计数器（编译时/运行时均生效）
    pub fn reset(self: *@This()) void {
        if (@inComptime()) {
            self.comptime_counter = 0;
        } else {
            self.runtime_counter.store(0, .SeqCst);
        }
    }
};
