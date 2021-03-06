import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
        )

    def forward(self, x):
        return x + self.main(x)

class GeneratorCNN(nn.Module):
    def __init__(self, input_channel, output_channel, conv_dims, deconv_dims, num_gpu):
        super(GeneratorCNN, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = conv_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        for out_dim in deconv_dims:
            self.layers.append(nn.ConvTranspose2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = out_dim

        self.layers.append(nn.ConvTranspose2d(prev_dim, output_channel, 4, 2, 1, bias=False))
        self.layers.append(nn.Tanh())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out

    def forward(self, x):
        return self.main(x)
class EncoderCNN_1(nn.Module):
    def __init__(self, input_channel, conv_dims, num_gpu):
        super(EncoderCNN_1, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = conv_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim
            
        for i in range(6):
            self.layers.append(ResidualBlock(dim_in=self.out_dim, dim_out=self.out_dim))

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out

    def forward(self, x):
        return self.main(x)
class EncoderCNN_2(nn.Module):
    def __init__(self, input_channel, conv_dims, num_gpu):
        super(EncoderCNN_2, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = conv_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out

    def forward(self, x):
        return self.main(x)             
class DecoderCNN(nn.Module):
    def __init__(self, input_channel, output_channel, deconv_dims, num_gpu):
        super(DecoderCNN, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []
        prev_dim = input_channel
        for out_dim in deconv_dims:
            self.layers.append(nn.ConvTranspose2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = out_dim

        self.layers.append(nn.ConvTranspose2d(prev_dim, output_channel, 4, 2, 1, bias=False))
        self.layers.append(nn.Tanh())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out

    def forward(self, x):
        return self.main(x)

class DiscriminatorCNN(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_dims, num_gpu):
        super(DiscriminatorCNN, self).__init__()
        self.num_gpu = num_gpu
        self.layers = []

        prev_dim = hidden_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in hidden_dims[1:]:
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_dim))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim
        #self.layers.append(nn.Conv2d(prev_dim, output_channel, 4, 1, 0, bias=False))
        #self.layers.append(nn.Linear(512*4*4, 1024))
        #self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Linear(prev_dim, output_channel))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x):
        out = x
        for index in range(len(self.layer_module)-2):
            layer = self.layer_module[index]
            out = layer(out)
        
        out=out.view(out.size(0), out.size(1),-1)
        #print(out.size())
        out=torch.mean(out,2)
        out=out.view(out.size(0), out.size(1))
        #print(out.size())
        #print out.size(3)
        #import IPython
        #IPython.embed()
        #out = self.layer_module[-4](out)
        #out = self.layer_module[-3](out)
        out = self.layer_module[-2](out)
        out = self.layer_module[-1](out)
        return out

    def forward(self, x):
        return self.main(x)

class GeneratorFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(GeneratorFC, self).__init__()
        self.layers = []

        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out

class DiscriminatorFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(DiscriminatorFC, self).__init__()
        self.layers = []

        prev_dim = input_size
        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim

        self.layers.append(nn.Linear(prev_dim, output_size))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(-1, 1)
