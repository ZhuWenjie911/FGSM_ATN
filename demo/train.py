import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch import optim
from demo.ZhuNet import Net
from demo.dateloader import return_data
import argparse
import torch

#parameters
epochs = 10
batch_size = 100
lr = 0.01

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--dset_dir', type=str, default='datasets')
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net().to(device)

opimizer = optim.Adam(net.parameters(),lr=lr)
loss_func = nn.CrossEntropyLoss()

data_load = return_data(args)
train_data = data_load['train']
test_data = data_load['test']

def traindemo():
    net.train()
    for epoch in range(epochs):
        count = 0
        for batch_id, (x, y) in enumerate(train_data):
            x_val = Variable(x).to(device)
            y_val = Variable(y).to(device)
            out = net(x_val)
            loss = loss_func(out, y_val)
            opimizer.zero_grad()
            loss.backward()
            opimizer.step()
            count += batch_size
            if count % 1000 == 0:
                print('epoch{} loss is {:.4f}'.format(count,loss.item()))

    torch.save({'net': net.state_dict()}, 'model_test.pt')

def testdemo():
    net.eval()
    state = torch.load('model_test.pt')
    net.load_state_dict(state['net'])
    eval_loss = 0
    eval_acc = 0
    for batch,data in enumerate(test_data):
        img, label = data
        # img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = net(img)
        loss = loss_func(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))


if __name__ == '__main__':
    traindemo()











