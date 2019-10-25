import argparse
import os
import itertools
# pickle用于序列化
import pickle
import imageio

# 绘图工具
import matplotlib.pyplot as plt

# nn里主要是一些网络结构
import torch
import torch.nn as nn
# nn.fuctional as F
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# datasets是包含几个常用视觉数据库,transoforms包含常用的图像处理操作
from torchvision import datasets, transforms


# G(z)
class Generator(nn.Module):
    # 实现GAN的生成器
    # 网络定义其实没什么，主要是看怎么训练的
    def __init__(self, input_size=32, n_class=10):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, input):
        # 把激活函数放到里面来
        x = F.leaky_relu(self.fc1(input), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)

        #x = F.tanh(self.fc4(x))
        #support torch1.1.0
        x = torch.tanh(self.fc4(x))

        # 输出大小等于D的输入
        return x


class Discriminator(nn.Module):
    # 实现GAN的判别器
    def __init__(self, input_size=32, n_class=10):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), negative_slope=0.2)
        # 预测时，每个参数都要×0.3，dropout的训练时间比没用的长
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)

        # 用sigmoid函数把结果变成概率
        #x = F.sigmoid(self.fc4(x))
        #support torch 1.1.0
        x = torch.sigmoid(self.fc4(x))

        return x


def parse_args():
    parser = argparse.ArgumentParser(description='Train args for GAN')
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--train_epoch', default=100, type=int)
    args = parser.parse_args()
    return args


# volatile=True就是不会被求导的意思，V0.4之后就已经没有了，用with no grad 代替
def train(args):
    # 5*5是图像中大小,fix代表什么？
    fixed_z = torch.randn((5 * 5, 100)).cuda()
    #fixed_z = Variable(fixed_z.cuda(), volatile=True)

    def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
        '''
        显示训练过程中的图像
        :param num_epoch:
        :param show:
        :param save:
        :param path:
        :param isFix:
        :return:
        '''
        z_ = torch.randn((5 * 5, 100)).cuda()
        #z_ = Variable(z_.cuda(), volatile=True)

        G.eval()
        with torch.no_grad():
            if isFix:
                test_imgs = G(fixed_z)
            else:
                test_imgs = G(z_)
        G.train()

        size_fig_grid = 5
        fig, ax = plt.subplot(size_fig_grid, size_fig_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_fig_grid), range(size_fig_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5 * 5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(test_imgs[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        if save:
            plt.savefig(path)

        # 显示
        if show:
            plt.show()
        else:
            plt.close()

    def show_train_loss(hist, show=False, save=False, path='train_hist.png'):
        '''
        显示训练过程中的loss图像
        :param hist:
        :param show:
        :param save:
        :param path:
        :return:
        '''
        x = range(len(hist['D_losses']))

        y1 = hist['D_losses']
        y2 = hist['G_lossed']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')

        # 设置横坐标、纵坐标
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()

    # 定义数据是怎么读入的
    # 定义图像预处理方式
    transform = transforms.Compose([
        # 把几个预处理操作写到一起
        # 先把图像格式由numpy转为tensor
        transforms.ToTensor(),
        # 图像标准化
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # 定义训练图像怎么载入
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True
    )

    # 定义网络,G的输入为什么是100
    G = Generator(input_size=100, n_class=28 * 28)  # 28?
    D = Discriminator(input_size=28 * 28, n_class=1)
    G.cuda()
    D.cuda()

    # 定义损失函数
    BCE_loss = nn.BCELoss()

    # 定义参数迭代器
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    if not os.path.isdir('MNIST_GAN_results'):
        os.mkdir('MNIST_GAN_results')
    if not os.path.isdir('MNIST_GAN_results/Random_results'):
        os.mkdir('MNIST_GAN_results/Random_results')
    if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
        os.mkdir('MNIST_GAN_results/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []

    # 开始训练
    for epoch in range(args.train_epoch):
        # 存放每一轮的损失
        D_losses = []
        G_losses = []
        # _ ?每一个batch
        for x_, _ in train_loader:
            # ---------------train discriminator D------
            # 不需要将两个batch的梯度混合起来累计
            D.zero_grad()

            # 真实图片
            x_ = x_.view(-1, 28 * 28)

            mini_batch = x_.size()[0]
            assert args.batch_size == mini_batch

            # 标签只需要是真或者假
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)

            # 丢到cuda里
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            D_result = D(x_)  # 输出结果是概率
            D_real_loss = BCE_loss(D_result, y_real_)
            D_real_score = D_result

            # ----------------train D---------------
            x_G = torch.randn((mini_batch, 100))
            x_G = Variable(x_G.cuda())
            G_result = G(x_G)  # 输出是一张图
            # 虽然计算了G，但还是在训练D

            # 讲G的输出图片送到D中判别是真还是假
            D_result = D(G_result)
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result

            # D的loss是两部分的loss之和
            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            # 把loss加到这个epoch的loss里
            D_losses.append(D_train_loss.item())

            # ----------------train G---------------
            G.zero_grad()

            # 为什么不把前面的G的计算结果拿过来呢
            x_G = torch.rand((mini_batch, 100))
            y_G = torch.ones(mini_batch)

            x_G, y_G = Variable(x_G.cuda()), Variable(y_G.cuda())
            G_result = G(x_G)
            D_result = D(G_result)
            G_train_loss = BCE_loss(D_result, y_G)

            # 反馈，计算下一个参数
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.item())  # 0?
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), args.train_epoch, torch.mean(torch.FloatTensor(D_losses)),
            torch.mean(torch.FloatTensor(G_losses))))
        path = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        path_fixed = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        show_result((epoch + 1), save=True, path=path, isFix=False)
        show_result((epoch + 1), save=True, path=path_fixed, isFix=True)

        # 每一个epoch之后绘制损失
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    # --------------------训练结束-----------------
    print('Training finish...')
    # 把训练结果存起来
    torch.save(G.state_dict(), "MNIST_GAN_results/generator_param.pkl")
    torch.save(D.state_dict(), "MNIST_GAN_results/discriminator_param.pkl")

    # 存储train_hist，需要用pickle进行序列化
    with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    # 把train_hist的结果显示出来，但是不能在线观看
    show_train_loss(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

    # 把图像存储为动图
    images = []
    for e in range(args.train_epoch):
        img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

# net.train和.eval采取不同的处理方式，比如batch Normalizaiton和dropout

if __name__ == '__main__':
    args = parse_args()
    train(args)
