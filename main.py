# coding: utf-8
# Implementing LeNet-5 Architecture On MNIST Dataset (GPU Implementation)
# 这里用GPU、LeNet-5网络结构给MNIST数据集做分类
import torch
# torch.multiprocessing.set_start_method("spawn")        # https://github.com/pytorch/pytorch/issues/3491#event-1326332533
import torch.nn
import torch.optim
import torch.nn.functional
import torchvision.datasets
import torchvision.transforms
import numpy as np
from models.LeNet5 import LeNet5
from matplotlib import pyplot
from sklearn.metrics import accuracy_score

def train():
    transformImg = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225])])
    train = torchvision.datasets.MNIST(root='D:\\github\\MNIST_mianjin\\data\\data',
                                       train=True,
                                       download=False,
                                       transform=transformImg)
    # train分成训练集和验证集（8：2）
    idx = list(range(len(train)))
    np.random.seed(1009)
    np.random.shuffle(idx) #（打乱顺序）
    train_idx = idx[: int(0.8 * len(idx))]
    valid_idx = idx[int(0.8 * len(idx)):]

    # 产生训练集和测试集样本
    train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # Load training and validation data based on above samples
    # Size of an individual batch during training and validation is 30
    # Both training and validation datasets are shuffled at every epoch by 'SubsetRandomSampler()'. Test set is not shuffled.
    train_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=train_set, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=valid_set, num_workers=4)

    net = LeNet5()
    net.cuda()
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    # 定义优化函数
    optimization = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 开始训练
    numEpochs = 20
    training_accuracy = []
    validation_accuracy = []

    for epoch in range(numEpochs):
        epoch_training_loss = 0.0
        num_batches = 0
        for batch_num, training_batch in enumerate(train_loader):
            inputs, labels = training_batch
            inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())
            optimization.zero_grad()
            forward_output = net(inputs)
            loss = loss_func(forward_output, labels)
            loss.backward()
            optimization.step()
            epoch_training_loss += loss.data
            num_batches += 1

        print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)

        # 计算训练集的准确率
        accuracy = 0.0
        num_batches = 0
        for batch_num, training_batch in enumerate(train_loader):  # 'enumerate' is a super helpful function
            num_batches += 1
            inputs, actual_val = training_batch
            predicted_val = net(torch.autograd.Variable(inputs.cuda()))
            # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function
            predicted_val = predicted_val.cpu().data.numpy()  # convert cuda() type to cpu(), then convert it to numpy
            predicted_val = np.argmax(predicted_val, axis=1)  # retrieved max_values along every row
            accuracy += accuracy_score(actual_val.numpy(), predicted_val)
        training_accuracy.append(accuracy / num_batches)

        # 计算验证集的准确率
        accuracy = 0.0
        num_batches = 0
        for batch_num, validation_batch in enumerate(valid_loader):  # 'enumerate' is a super helpful function
            num_batches += 1
            inputs, actual_val = validation_batch
            # perform classification
            predicted_val = net(torch.autograd.Variable(inputs.cuda()))
            # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function
            predicted_val = predicted_val.cpu().data.numpy()  # convert cuda() type to cpu(), then convert it to numpy
            predicted_val = np.argmax(predicted_val, axis=1)  # retrieved max_values along every row
            # accuracy
            accuracy += accuracy_score(actual_val.numpy(), predicted_val)
        validation_accuracy.append(accuracy / num_batches)

    # 保存模型
    models_name = 'checkpoints/LeNet5.pth'
    torch.save(net.state_dict(), models_name)

    epochs = list(range(numEpochs))

    # plotting training and validation accuracies
    fig1 = pyplot.figure()
    pyplot.plot(epochs, training_accuracy, 'r')
    pyplot.plot(epochs, validation_accuracy, 'g')
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Accuracy")
    pyplot.show(fig1)

def test():
    transformImg = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                        (0.5, 0.5, 0.5))])
    test = torchvision.datasets.MNIST(root='D:\\github\\MNIST_mianjin\\data\\data',
                                      train=False,
                                      download=True,
                                      transform=transformImg)
    test_loader = torch.utils.data.DataLoader(test, num_workers=4)

    net = LeNet5()
    net.cuda()

    # 加载模型参数
    name = 'checkpoints/LeNet5_1.pth'
    net.load_state_dict(torch.load(name))
    print(net)
    # 在测试集上验证模型准确率
    correct = 0
    total = 0
    for test_data in test_loader:
        total += 1
        inputs, actual_val = test_data
        predicted_val = net(torch.autograd.Variable(inputs.cuda()))
        predicted_val = predicted_val.cpu().data # 从GPU tensor转为CPU tensor
        max_score, idx = torch.max(predicted_val, 1)
        correct += (idx == actual_val).sum()
    correct = correct.numpy()
    print("Classifier Accuracy: ", correct / total * 100)


if __name__=="__main__":
    # train()
    test()

