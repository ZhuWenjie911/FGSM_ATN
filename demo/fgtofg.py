from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from demo.ZhuNet import Net
from demo.dateloader import return_data
import argparse
from torch.autograd import Variable

epsilons = [0.5]
pretrained_model = "model_test.pt"
use_cuda=True

epochs = 2
batch_size = 1
lr = 0.01

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--dset_dir', type=str, default='datasets')
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = Net().to(device)

opimizer = optim.Adam(net.parameters(),lr=lr)
loss_func = nn.CrossEntropyLoss()

data_load = return_data(args)
train_data = data_load['train']
test_data = data_load['test']

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image

def target_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image - epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image

def test(epsilon,target):


    ori_example = []
    target_example = []
    ori = []
    tar = []


    net.eval()
    state = torch.load('model_test.pt')
    net.load_state_dict(state['net'])
    eval_loss = 0
    eval_acc = 0
    testsum = 0
    sum = 0
    for batch, datas in enumerate(test_data):
        testsum += 1
        img, label = datas

        # img = img.view(img.size(0), -1)
        # img = Variable(img)

        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            target = Variable(torch.tensor([target])).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
            target = Variable(torch.tensor([target]))

        # print(label)
        # print(target)
        # print(img)


        img.requires_grad = True



        out = net(img)

        init_pred = out.max(1, keepdim=True)[1]

        if init_pred.item() != label.item() :
            continue


        loss = loss_func(out, label)
        net.zero_grad()
        loss.backward()
        data_grad = img.grad

        perturbed_data = fgsm_attack(img, epsilon, data_grad)

        output = net(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        # print(final_pred)
        # print("\n")
        if final_pred.item() != label.item():

            # perturbed_data.requires_grad = True
            ori.append(label)
            tar.append(final_pred)
            ori_example.append(img)
            target_example.append(perturbed_data)
            sum += 1
            print(sum)
            print(testsum)
            # print(img)
            # print(perturbed_data)
            # print(perturbed_data - img)

        if sum == 100:
            return ori_example,ori,target_example,tar


    return ori_example,ori,target_example,tar

def test2(epsilon,ori_example,ori,target_example,tar):
    sum = 0
    result = []
    net.eval()
    state = torch.load('model_test.pt')
    net.load_state_dict(state['net'])

    for i in range(len(target_example)):

        # img = img.view(img.size(0), -1)
        # img = Variable(img)


        if torch.cuda.is_available():
            img = Variable(target_example[i]).cuda()
            label = Variable(torch.tensor([tar[i]])).cuda()
            ori_lb = Variable(torch.tensor([ori[i]])).cuda()
            ori_img = Variable(ori_example[i]).cuda()
        else:
            img = Variable(target_example[i])
            label = Variable(torch.tensor([tar[i]]))
            ori_lb = Variable(torch.tensor([ori[i]]))
            ori_img = Variable(ori_example[i])

        # print(label)
        # print(ori_lb)

        img.requires_grad = True



        out = net(img)

        init_pred = out.max(1, keepdim=True)[1]

        if init_pred.item() != label.item():
            continue


        loss = loss_func(out, ori_lb)
        net.zero_grad()
        loss.backward()
        data_grad = img.grad

        perturbed_data = target_attack(img, epsilon, data_grad)

        output = net(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        # print(final_pred)
        # print("\n")
        if final_pred.item() == ori_lb:
            sum += 1
            print(sum)
            print(i)
            result.append(perturbed_data - ori_img)

            print(img - ori_img)
            print(perturbed_data - ori_img)
    return result


if __name__ == '__main__':
    ori_examples = []
    target_example = []
    ori = []
    target = 7



    # Run test for each epsilon
    ori_examples,ori,target_example,tar = test(0.2,target)
    result = test2(0.2,ori_examples,ori,target_example,tar)
    # print(result)




