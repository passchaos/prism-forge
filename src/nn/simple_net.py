import numpy as np
import time

# 辅助函数：数值稳定的softmax
def softmax(x):
    # 减去最大值避免数值溢出
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 辅助函数：交叉熵损失（带softmax，数值稳定）
def cross_entropy_loss(logits, y_true):
    batch_size = logits.shape[0]
    # 直接用logits计算，避免先算softmax再取log（防止数值下溢）
    logits_max = np.max(logits, axis=1, keepdims=True)
    log_probs = logits - logits_max - np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True))
    # 只取真实标签对应的log概率
    loss = -np.sum(log_probs[np.arange(batch_size), y_true]) / batch_size
    return loss

# 简单网络类
class SimpleNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 初始化参数（使用小随机数）
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        # 前向传播：linear -> relu -> linear
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2  # 返回logits（不做softmax，留给损失函数）

    def compute_loss(self, x, y):
        logits = self.forward(x)
        return cross_entropy_loss(logits, y)

    def backward(self, x, y):
        # 反向传播计算梯度
        batch_size = x.shape[0]
        logits = self.forward(x)

        # 1. softmax+cross_entropy的梯度（核心：logits的梯度 = softmax - one_hot）
        y_one_hot = np.zeros_like(logits)
        y_one_hot[np.arange(batch_size), y] = 1
        dz2 = (softmax(logits) - y_one_hot) / batch_size

        # 2. W2和b2的梯度
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        # 3. 反向传播到ReLU层
        da1 = np.dot(dz2, self.W2.T)
        # ReLU的梯度：z1>0时为1，否则为0
        dz1 = da1 * (self.z1 > 0)

        # 4. W1和b1的梯度
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }

# 数值梯度计算函数（中心差分）
def compute_numerical_gradient(net, x, y, param_name, h=1e-5):
    """
    计算指定参数的数值梯度
    param_name: 'W1', 'b1', 'W2', 'b2'
    h: 扰动值（通常取1e-5~1e-4）
    """
    param = getattr(net, param_name)
    grad = np.zeros_like(param)

    # 遍历参数的每个元素，逐个扰动
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        # 保存原始值
        original_val = param[idx]

        # f(x+h)
        param[idx] = original_val + h
        loss_plus = net.compute_loss(x, y)

        # f(x-h)
        param[idx] = original_val - h
        loss_minus = net.compute_loss(x, y)

        # 中心差分计算梯度
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

        # 恢复原始值
        param[idx] = original_val
        it.iternext()

    return grad

# 测试代码
if __name__ == "__main__":
    # 1. 初始化参数
    input_dim = 800
    hidden_dim = 50
    output_dim = 10
    batch_size = 50

    # 2. 创建数据
    x = np.random.randn(batch_size, input_dim)  # 输入
    y = np.random.randint(0, output_dim, size=batch_size)  # 标签
    print(x.dtype)

    # 3. 创建网络
    net = SimpleNet(input_dim, hidden_dim, output_dim)

    begin = time.time()
    for i in range(10000):
        net.compute_loss(x, y)
    end = time.time()
    elapsed = end - begin
    print(f"计算时间：{elapsed:.6f}秒")

    # 4. 计算解析梯度（反向传播）
    # analytic_grads = net.backward(x, y)

    # # 5. 计算数值梯度
    # numerical_grads = {}
    # for param_name in ['W1', 'b1', 'W2', 'b2']:
    #     start_time = time.time()
    #     numerical_grads[param_name] = compute_numerical_gradient(net, x, y, param_name)
    #     end_time = time.time()
    #     print(f"{param_name}计算时间：{end_time - start_time:.6f}秒")

    # # 6. 对比梯度（计算相对误差）
    # print("梯度对比（相对误差）：")
    # for param_name in ['W1', 'b1', 'W2', 'b2']:
    #     analytic = analytic_grads[param_name]
    #     numerical = numerical_grads[param_name]
    #     # 计算相对误差（避免除0）
    #     abs_diff = np.abs(analytic - numerical)
    #     rel_error = abs_diff / (np.maximum(np.abs(analytic) + np.abs(numerical), 1e-8))

    #     mean_abs_error = np.mean(abs_diff)
    #     mean_rel_error = np.mean(rel_error)
    #     print(f"{param_name}: abs_error= {mean_abs_error:.15f} rel_error= {mean_rel_error:.15f}")
