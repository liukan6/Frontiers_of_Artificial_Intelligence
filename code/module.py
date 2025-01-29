import numpy as np

class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dtype = None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype
        
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(dtype)
        self.bias = np.random.randn(out_channels).astype(dtype)

    def forward(self, x):
        self.x = x  # 添加这一行
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        y = np.zeros((N, self.out_channels, H_out, W_out), dtype=self.dtype)

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        y[n, c_out, h, w] = np.sum(
                            x[n, :, h_start:h_end, w_start:w_end] * self.weight[c_out, :, :, :]
                        ) + self.bias[c_out]
        return y

    def backward(self, dy, lr):
        N, C, H_out, W_out = dy.shape
        _, _, H, W = self.x.shape
        dx = np.zeros_like(self.x)  # 使用 self.x
        self.w_grad = np.zeros_like(self.weight)
        self.b_grad = np.zeros_like(self.bias)

        if self.padding > 0:
            self.x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        padded_dx = np.pad(dx, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        if h_end <= self.x.shape[2] and w_end <= self.x.shape[3]:
                            padded_dx[n, :, h_start:h_end, w_start:w_end] += dy[n, c_out, h, w] * self.weight[c_out, :, :, :]
                            self.w_grad[c_out, :, :, :] += dy[n, c_out, h, w] * self.x[n, :, h_start:h_end, w_start:w_end]
                self.b_grad[c_out] += np.sum(dy[n, c_out])
        
        if self.padding > 0:
            dx = padded_dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = padded_dx

        self.weight -= lr * self.w_grad
        self.bias -= lr * self.b_grad
        return dx
    
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
      
    def backward(self, dy):
        return dy * (self.x > 0)
       
class Tanh:
    def forward(self, x):
        self.x = x
        return np.tanh(x)
       
    def backward(self, dy):
        return dy * (1 - np.tanh(self.x) ** 2)
        
class Sigmoid:
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
       
    def backward(self, dy):
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        return dy * sigmoid_x * (1 - sigmoid_x)

class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 减去最大值以防止溢出
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.probs

    def backward(self, dy):
        # 初始化dx的形状
        dx = np.zeros_like(dy)
        for i, (dy_i, prob_i) in enumerate(zip(dy, self.probs)):
            # Jacobian矩阵的计算
            prob_i = prob_i.reshape(-1, 1)
            jacobian = np.diagflat(prob_i) - np.dot(prob_i, prob_i.T)
            dx[i] = np.dot(jacobian, dy_i)
        return dx

       
class MaxPool2d:
    def __init__(self, kernel_size: int, stride = None, padding = 0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        y = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        self.x = x

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        y[n, c, h, w] = np.max(x[n, c, h_start:h_end, w_start:w_end])
        return y

    def backward(self, dy):
        N, C, H_out, W_out = dy.shape
        dx = np.zeros_like(self.x)
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        x_slice = self.x[n, c, h_start:h_end, w_start:w_end]
                        mask = (x_slice == np.max(x_slice))
                        dx[n, c, h_start:h_end, w_start:w_end] += dy[n, c, h, w] * mask
        return dx

     
class AvgPool2d:
    def __init__(self, kernel_size: int, stride = None, padding = 0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        y = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        self.x = x

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        y[n, c, h, w] = np.mean(x[n, c, h_start:h_end, w_start:w_end])
        return y

    def backward(self, dy):
        N, C, H_out, W_out = dy.shape
        dx = np.zeros_like(self.x)
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        dx[n, c, h_start:h_end, w_start:w_end] += dy[n, c, h, w] / (self.kernel_size * self.kernel_size)
        return dx
           

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features) if bias else None

    def forward(self, x):
        self.x = x
        y = x @ self.weight.T
        if self.bias is not None:
            y += self.bias
        return y

    def backward(self, dy, lr):
        # 计算梯度
        self.w_grad = dy.T @ self.x
        if self.bias is not None:
            self.b_grad = np.sum(dy, axis=0)
        
        # 计算输入的梯度
        dx = dy @ self.weight
        
        # 更新参数
        self.weight -= lr * self.w_grad
        if self.bias is not None:
            self.bias -= lr * self.b_grad

        return dx

        

class CrossEntropyLoss:
    def __call__(self, x, label):
        N = x.shape[0]
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.probs = probs
        self.label = label
        loss = -np.sum(np.log(probs[np.arange(N), label])) / N
        return loss
    # SGD; 再写个AdmaW
    def backward(self):
        N = self.label.shape[0]
        dx = self.probs.copy()
        dx[np.arange(N), self.label] -= 1
        dx /= N
        return dx
