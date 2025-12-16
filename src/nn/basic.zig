const Tensor = @import("../Tensor.zig");

pub fn numericalDiff(x: Tensor, comptime T: type, f: fn (T) T) T {
    const h = 1e-4;

    var grad = try Tensor.zerosLike(x.allocator, x);
    
    var x_iter = try x.dataIter();
    defer x_iter.deinit();
    
    while (x_iter.next()) |idx| {
        
    }
    
    const fx = f(x - h);
    const fxh = f(x + h);
    return (fxh - fx) / (h + h);
}
