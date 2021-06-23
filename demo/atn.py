import torch


nn = torch.nn

class GATN_Conv(nn.Module):
    '''
    USAGE: atn = GATN_Conv()
    This is ATN_a that takes also gradient as input and use convolutional layer
    CONTRIBUTER: henryliu, 07.25
    '''

    def __init__(self, beta=2, innerlayer=100, width=28, channel=1):
        '''
        This is a simple net of (width * width)->FC( innerlayer )->FC(width * width)-> sigmoid(grad + output)
        '''
        super().__init__()
        self.beta = beta
        self.channel = channel
        self.width = width
        self.layer1_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(1, 8, 5, 1), # use padding = 0
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (10, 14, 14)
        )
        self.layer2_conv = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(1, 8, 5, 1), # use padding = 0
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (10, 14, 14)
        )
        self.layer3 = nn.Sequential(
            nn.Linear( 2304, width * width * channel),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(width * width * channel, width * width * channel)
        )
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.torch.nn.init.xavier_normal_(m.weight)
        self.out_image = nn.Sigmoid()

    def forward(self, x_image, x_grad):
        self.batch_size = x_image.size(0)
        x1 = self.layer1_conv(x_image).view(self.batch_size, -1)
        x2 = self.layer2_conv(x_grad).view(self.batch_size, -1)
        x = torch.cat((x1, x2), dim=1) # [D, 10 * width/4 *width /4 * 2]
        x_image = x_image.view(x_image.size(0), -1)
        x_grad = x_grad.view(x_grad.size(0), -1)
        x = self.layer3(x)
        x = self.layer4(x + x_grad) # x is perturbation
        x = self.out_image( (x + x_image-0.5)*5 ) # adding pertubation and norm
        return x.view(x.size(0), self.channel, self.width, self.width)