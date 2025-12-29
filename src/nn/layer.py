import numpy as np

def softmax(x):
    if x.ndim == 2:
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis = 1, keepdims=True)
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_loss(logits, y_true):
    batch_size = logits.shape[0]
    # 直接用logits计算，避免先算softmax再取log（防止数值下溢）

    logits_max = np.max(logits, axis=1, keepdims=True)
    log_probs = logits - logits_max - np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True))
    # 只取真实标签对应的log概率
    loss = -np.sum(log_probs[np.arange(batch_size), y_true]) / batch_size
    return loss

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        it.iternext()
    return grad

class TestNet:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        print(f"x= {x} w= {self.w}")
        self.z1 = np.dot(x, self.w) + self.b
        self.a1 = np.maximum(0, self.z1)

        return self.a1

    def loss(self, x, y):
        logits = self.forward(x)
        return cross_entropy_loss(logits, y)

    def backward(self, x, t):
        batch_size = x.shape[0]
        logits = self.forward(x)

        # 1. softmax+cross_entropy的梯度（核心：logits的梯度 = softmax - one_hot）
        y_one_hot = np.zeros_like(logits)
        y_one_hot[np.arange(batch_size), t] = 1
        dz2 = (softmax(logits) - t) / batch_size

        # ReLU的梯度：z1>0时为1，否则为0
        dz1 = dz2 * (self.z1 > 0)

        # 4. W1和b1的梯度
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        return {
            'w': dW1, 'b': db1,
        }

    def numerical_gradient(self, x, t):
        w = numerical_gradient(lambda w: self.loss(x, t), self.w)
        b = numerical_gradient(lambda w: self.loss(x, t), self.b)

        grads = {}
        grads['w'] = w
        grads['b'] = b
        return grads

if __name__ == '__main__':
    w = np.arange(10).reshape(2, 5)
    b = np.arange(5).reshape(1, 5);

    # affine = Affine(w, b)
    # relu = Relu()
    # softmax_with_loss = SoftmaxWithLoss()

    x = np.arange(6).reshape(3, 2)
    # t = np.array([[0, 0, 0, 0, 1], [0,0,0,1,0], [1, 0, 0, 0, 0]])
    t = np.array([4, 3, 0])

    net = TestNet(w, b)
    grads1 = net.numerical_gradient(x, t)
    grads2 = net.backward(x, t)
    print(grads1)
    print(grads2)
