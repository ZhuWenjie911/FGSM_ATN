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

epsilons = [0, .05, .1, .15, .2, .25, .3]
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
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test(epsilon):
    correct = 0
    adv_examples = []



    net.eval()
    state = torch.load('model_test.pt')
    net.load_state_dict(state['net'])
    eval_loss = 0
    eval_acc = 0
    for batch, datas in enumerate(test_data):
        img, label = datas
        # img = img.view(img.size(0), -1)
        # img = Variable(img)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # print(label)

        img.requires_grad = True


        out = net(img)

        init_pred = out.max(1, keepdim=True)[1]

        if init_pred.item() != label.item():
            continue


        loss = loss_func(out, label)
        net.zero_grad()
        loss.backward()
        data_grad = img.grad.data

        perturbed_data = fgsm_attack(img, epsilon, data_grad)

        output = net(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == label.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_data))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_data), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

if __name__ == '__main__':
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(eps)
        accuracies.append(acc)
        examples.append(ex)
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

