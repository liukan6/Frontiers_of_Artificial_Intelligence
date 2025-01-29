import numpy as np
from module import Conv2d, Sigmoid, MaxPool2d, AvgPool2d, Linear, ReLU, Tanh , Softmax, CrossEntropyLoss
import struct
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

class LeNet5:
    def __init__(self):
        self.conv1 = Conv2d(1, 6, 5, 1, 2)
        self.relu1 = Sigmoid()
        self.pool1 = AvgPool2d(2)
        self.conv2 = Conv2d(6, 16, 5)
        self.relu2 = ReLU()
        self.pool2 = AvgPool2d(2)
        self.fc1 = Linear(16*5*5, 120)
        self.relu3 = Sigmoid()
        self.fc2 = Linear(120, 84)
        self.relu4 = Sigmoid()
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        x = self.relu4.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, dy, lr):
        dy = self.fc3.backward(dy, lr)
        dy = self.relu4.backward(dy)
        dy = self.fc2.backward(dy, lr)
        dy = self.relu3.backward(dy)
        dy = self.fc1.backward(dy, lr)
        dy = dy.reshape(-1, 16, 5, 5)
        dy = self.pool2.backward(dy)
        dy = self.relu2.backward(dy)
        dy = self.conv2.backward(dy, lr)
        dy = self.pool1.backward(dy)
        dy = self.relu1.backward(dy)
        dy = self.conv1.backward(dy, lr)

def save_model(model, path):
    params = {
        "conv1_weight": model.conv1.weight,
        "conv1_bias": model.conv1.bias,
        "conv2_weight": model.conv2.weight,
        "conv2_bias": model.conv2.bias,
        "fc1_weight": model.fc1.weight,
        "fc1_bias": model.fc1.bias,
        "fc2_weight": model.fc2.weight,
        "fc2_bias": model.fc2.bias,
        "fc3_weight": model.fc3.weight,
        "fc3_bias": model.fc3.bias
    }
    np.save(path, params)

def load_model(model, path):
    params = np.load(path, allow_pickle=True).item()
    model.conv1.weight = params["conv1_weight"]
    model.conv1.bias = params["conv1_bias"]
    model.conv2.weight = params["conv2_weight"]
    model.conv2.bias = params["conv2_bias"]
    model.fc1.weight = params["fc1_weight"]
    model.fc1.bias = params["fc1_bias"]
    model.fc2.weight = params["fc2_weight"]
    model.fc2.bias = params["fc2_bias"]
    model.fc3.weight = params["fc3_weight"]
    model.fc3.bias = params["fc3_bias"]

if __name__ == '__main__':
    batch_size = 100
    train_images, train_labels = load_mnist("E:\Repository\Frontiers_of_Artificial_Intelligence\MNIST_Dataset", kind="train")
    test_images, test_labels = load_mnist("E:\Repository\Frontiers_of_Artificial_Intelligence\MNIST_Dataset", kind="t10k")

    train_images = train_images.astype(np.float32) / 255
    test_images = test_images.astype(np.float32) / 255
    train_images = train_images.reshape(-1, 1, 28, 28)
    test_images = test_images.reshape(-1, 1, 28, 28)

    model = LeNet5()
    model_path = "model_epoch_10.npy"

    # 如果存在已保存的模型参数，则加载它们
    if os.path.exists(model_path):
        load_model(model, model_path)
        print(f"Loaded model parameters from {model_path}")

    loss_fn = CrossEntropyLoss()
    lr = 0.01
    epochs = 10

    writer = SummaryWriter(log_dir=f'runs/mnist_experiment_1_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    for epoch in range(epochs):
        running_loss = 0.0
        for i in tqdm(range(0, len(train_images), batch_size), desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch"):
            batch_end = i + batch_size if i + batch_size < len(train_images) else len(train_images)
            x_batch = train_images[i:batch_end]
            label_batch = train_labels[i:batch_end]

            out = model.forward(x_batch)
            loss = loss_fn(out, label_batch)
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_images) // batch_size + i // batch_size)
            
            dy = loss_fn.backward()
            model.backward(dy, lr)

        avg_loss = running_loss / (len(train_images) // batch_size)
        writer.add_scalar('Loss/avg_train', avg_loss, epoch)
        tqdm.write(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')

        correct = 0
        for i in tqdm(range(0, len(test_images), batch_size), desc=f"Epoch {epoch+1}/{epochs} - Testing", unit="batch"):
            batch_end = i + batch_size if i + batch_size < len(test_images) else len(test_images)
            x_batch = test_images[i:batch_end]
            label_batch = test_labels[i:batch_end]

            out = model.forward(x_batch)
            preds = np.argmax(out, axis=1)
            correct += np.sum(preds == label_batch)

        accuracy = correct / len(test_images)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        tqdm.write(f'Epoch {epoch+1}/{epochs}, Accuracy: {accuracy}')

        # 保存模型
        save_model(model, model_path)

        # 记录模型参数
        for name, param in [("conv1_weight", model.conv1.weight), ("conv1_bias", model.conv1.bias),
                            ("conv2_weight", model.conv2.weight), ("conv2_bias", model.conv2.bias),
                            ("fc1_weight", model.fc1.weight), ("fc1_bias", model.fc1.bias),
                            ("fc2_weight", model.fc2.weight), ("fc2_bias", model.fc2.bias),
                            ("fc3_weight", model.fc3.weight), ("fc3_bias", model.fc3.bias)]:
            writer.add_histogram(name, param, epoch)

    writer.close()