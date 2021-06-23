import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch import optim
from demo.ZhuNet import Net
from demo.dateloader import return_data
from demo.atn import GATN_Conv
import argparse
import torch


def lossY_fn(y_now,  target):
    '''
        USAGE: returns MSE loss between y_now and target
        all y should be [D, 10] tensor, target is proposed class
        target should be an int
    '''
    #y_now = sigmoid_norm(y_now)
    #y_origin = sigmoid_norm(y_origin)
    #y_reranked_target = reranking(y_origin, target, alpha)
    y_target = torch.zeros_like(y_now)
    # print(y_target)
    y_target[:, target] = 1
    # print(y_target)
    # print(y_target)
    #KLloss_fn = nn.KLDivLoss()
    #lossY = KLloss_fn(torch.log(y_now), y_target)
    MSELoss_fn = nn.MSELoss()
    lossY = MSELoss_fn(y_now, y_target)
    return lossY

def sigmoid_norm(y):
    '''
    USAGE: y must be a [D,class_size] size tensor
    This function gives whatever input and norm it to possibility using sigmoid_norm
    '''
    y = y.sigmoid()
    y = y / y.sum(1).view(y.size(0), -1)
    return y

def accuracy(y_pred, y_label, target):
    '''
    USAGE: returns the accuracy/target_rate based on y_pred in shape [D, 10] and y_label in shape [D].
    '''
    _, predicted = torch.max(y_pred, 1)
    total = y_label.size(0)
    correct = (predicted == y_label).sum().item()
    accuracy = correct/total
    non_target_idx = (y_label != target)
    targetotal = (predicted[non_target_idx] == target).sum().item()
    targetrate = targetotal / non_target_idx.sum().item()
    return accuracy, targetrate

def cal_grad_target(X, cnn_model, target):
    '''
    USAGE: This function calculate target gradient with respect to target output probability
    (instead of loss)
    (they are actually the same)
    '''
    x_image = X.detach()
    x_image.requires_grad_(True)
    out = cnn_model(x_image)
    target_out = out[:, target]
    target_out.backward(torch.ones_like(target_out))
    return x_image.grad

def train(target):
    epochs = 5
    batch_size = 100
    lr = 0.001
    lossX_fn = nn.MSELoss()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='datasets')
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net().to(device)
    atn = GATN_Conv().to(device)

    net.eval()
    state = torch.load('model_test.pt')
    net.load_state_dict(state['net'])

    opimizer = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    data_load = return_data(args)
    train_data = data_load['train']
    test_data = data_load['test']


    atn.train()
    for epoch in range(epochs):
        count = 0
        for batch_id, (x, y) in enumerate(train_data):
            x_val = Variable(x).to(device)
            y_val = Variable(y).to(device)

            x_grad = cal_grad_target(
                x_val, net, target)
            x_adv = atn(x_val,x_grad)
            y_now = net(x_adv)
            lossX = lossX_fn(x_adv, x_val)
            lossY = lossY_fn(y_now,target)
            loss = lossX * atn.beta + lossY
            opimizer.zero_grad()
            loss.backward()
            opimizer.step()
            count += batch_size
            if count % 1000 == 0:

                x_test_grad = cal_grad_target(x_val, net, target)
                x_adv_test = atn(x_val, x_test_grad)
                y_pred = net(x_adv_test)
                acc, targetrate = accuracy(y_pred, y_val, target)
                print('Epoch:', epoch, '|Step:', count, '|loss Y:%.4f' %
                      lossY.item(), '|image norm:%.4f' % lossX, '|test accuracy:%.4f' % acc,
                      '|target rate:%.4f' % targetrate)
                if lossX < 0.011 and acc < 0.23 and targetrate > 0.86:
                    torch.save({'atn': atn.state_dict()}, 'atn_model.pt')
                    print('model saved')
                    return

                # self adjustment of parameter using threshold
                if lossX >= 0.03:
                    atn.beta *= 1.15
                elif lossX >= 0.015:
                    atn.beta *= 1.05
                elif lossX >= 0.01:
                    atn.beta *= 1.02

                if acc >= 0.4:
                    atn.beta /= 1.1
                elif acc >= 0.3:
                    atn.beta /= 1.05
                elif acc >= 0.20:
                    atn.beta /= 1.01
                print('epoch{} loss is {:.4f}'.format(batch_size, lossY.item()))

    torch.save({'atn': atn.state_dict()}, 'atn_model.pt')
    print('model saved')

if __name__ == '__main__':
    train(7)