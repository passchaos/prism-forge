import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        print(f"dout: {dout}")
        print(f"w_t: {self.W.T}")
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        print(f"y: {self.y}")
        print(f"t: {self.t}")
        dx = (self.y - self.t) / batch_size
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    print(f"batch_size: {batch_size} t: {t}")
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

if __name__ == '__main__':
    w = np.arange(10).reshape(2, 5)
    b = np.arange(5).reshape(1, 5);

    affine = Affine(w, b)
    relu = Relu()
    softmax_with_loss = SoftmaxWithLoss()

    x = np.arange(6).reshape(3, 2)
    t = np.array([[0, 0, 0, 0, 1], [0,0,0,1,0], [1, 0, 0, 0, 0]])
    print(f"input: {x}")
    f0 = affine.forward(x)
    f1 = relu.forward(f0)
    f2 = softmax_with_loss.forward(f1, t)
    print(f"f1: {f1}")
    print(f"f2: {f2}")

    b2 = softmax_with_loss.backward()
    print(f"b2: {b2}")
    b1 = relu.backward(b2)
    print(f"b1: {b1}")
    b0 = affine.backward(b1)

    print(f"b0: {b0}")

# class SoftmaxWithLoss:
#     def __init__(self):
#         self.loss = None
#         self.y = None
#         self.t = None

#     def forward(self, x, t):
#         self.t = t
#         self.y = softmax(x)
#         self.loss = cross_entropy_error(self.y, self.t)
#         return self.loss

#     def backward(self, dout=1):
#         batch_size = self.t.shape[0]
#         dx = (self.y - self.t) / batch_size
#         return dx
