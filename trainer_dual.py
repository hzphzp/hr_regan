from __future__ import print_function

import os
from glob import glob
from tqdm import trange
from itertools import chain

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

from models import *
from data_loader import get_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    def __init__(self, config, a_data_loader, b_data_loader):
        self.config = config

        self.a_data_loader = a_data_loader
        self.b_data_loader = b_data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.loss = config.loss
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.cnn_type = config.cnn_type

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.build_model()

        if self.num_gpu == 1:
            self.E_AB.cuda()
            self.E_BA.cuda()
            self.D_AB.cuda()
            self.D_BA.cuda()
            self.D_A.cuda()
            self.D_B.cuda()

        elif self.num_gpu > 1:
            self.E_AB = nn.DataParallel(self.E_AB.cuda(),device_ids=range(self.num_gpu))
            self.E_BA = nn.DataParallel(self.E_BA.cuda(),device_ids=range(self.num_gpu))
            self.D_AB = nn.DataParallel(self.D_AB.cuda(),device_ids=range(self.num_gpu))
            self.D_BA = nn.DataParallel(self.D_BA.cuda(),device_ids=range(self.num_gpu))
            self.D_A = nn.DataParallel(self.D_A.cuda(),device_ids=range(self.num_gpu))
            self.D_B = nn.DataParallel(self.D_B.cuda(),device_ids=range(self.num_gpu))

        if self.load_path:
            self.load_model()

    def build_model(self):
        if self.dataset == 'toy':
            self.G_AB = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)
            self.G_BA = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)

            self.D_A = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
            self.D_B = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
        else:
            a_height, a_width, a_channel = self.a_data_loader.shape
            b_height, b_width, b_channel = self.b_data_loader.shape

            if self.cnn_type == 0:
                #conv_dims, deconv_dims = [64, 128, 256, 512], [512, 256, 128, 64]
                conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
            elif self.cnn_type == 1:
                #conv_dims, deconv_dims = [32, 64, 128, 256], [256, 128, 64, 32]
                conv_dims, deconv_dims = [32, 64, 128, 256], [128, 64, 32]
            else:
                raise Exception("[!] cnn_type {} is not defined".format(self.cnn_type))
            
            self.E_AB = EncoderCNN_1(
                    a_channel, conv_dims, self.num_gpu)
            self.E_BA = EncoderCNN_2(
                    b_channel, conv_dims, self.num_gpu)
            self.D_AB = DecoderCNN(
                    conv_dims[-1], b_channel, deconv_dims, self.num_gpu)
            self.D_BA = DecoderCNN(
                    conv_dims[-1], a_channel, deconv_dims, self.num_gpu)
            '''
            self.G_AB = GeneratorCNN(
                    a_channel, b_channel, conv_dims, deconv_dims, self.num_gpu)
            self.G_BA = GeneratorCNN(
                    b_channel, a_channel, conv_dims, deconv_dims, self.num_gpu)
            '''

            self.D_A = DiscriminatorCNN(
                    a_channel, 1, conv_dims, self.num_gpu)
            self.D_B = DiscriminatorCNN(
                    b_channel, 1, conv_dims, self.num_gpu)

            self.E_AB.apply(weights_init)
            self.E_BA.apply(weights_init)
            self.D_AB.apply(weights_init)
            self.D_BA.apply(weights_init)

            self.D_A.apply(weights_init)
            self.D_B.apply(weights_init)

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'E_AB_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [int(os.path.basename(path.split('.')[0].split('_')[-1])) for path in paths]
        self.start_step = 55999#max(idxes)#9999

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        E_AB_filename = '{}/E_AB_{}.pth'.format(self.load_path, self.start_step)
        self.E_AB.load_state_dict(torch.load(E_AB_filename, map_location=map_location))
        self.E_BA.load_state_dict(
            torch.load('{}/E_BA_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))
            
        self.D_AB.load_state_dict(
            torch.load('{}/D_AB_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))
        self.D_BA.load_state_dict(
            torch.load('{}/D_BA_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

        self.D_A.load_state_dict(
            torch.load('{}/D_A_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))
        self.D_B.load_state_dict(
            torch.load('{}/D_B_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

        print("[*] Model loaded: {}".format(E_AB_filename))

    def train(self):
        d = nn.MSELoss()
        bce = nn.BCELoss()

        real_label = 1
        fake_label = 0

        real_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = real_tensor.data.fill_(real_label)

        fake_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = fake_tensor.data.fill_(fake_label)

        if self.num_gpu > 0:
            d.cuda()
            bce.cuda()

            real_tensor = real_tensor.cuda()
            fake_tensor = fake_tensor.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        optimizer_d = optimizer(
            chain(self.D_B.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_g = optimizer(
            chain(self.E_AB.parameters(), self.D_AB.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2))

        A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
        valid_x_A, valid_x_B = self._get_variable(A_loader.next()), self._get_variable(B_loader.next())

        vutils.save_image(valid_x_A.data*0.5+0.5, '{}/valid_x_A.png'.format(self.model_dir))
        vutils.save_image(valid_x_B.data*0.5+0.5, '{}/valid_x_B.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            try:
                x_A, x_B = A_loader.next(), B_loader.next()
            except StopIteration:
                A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
                x_A, x_B = A_loader.next(), B_loader.next()
            if x_A.size(0) != x_B.size(0):
                print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                continue

            x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)

            batch_size = x_A.size(0)
            real_tensor.data.resize_(batch_size).fill_(real_label)
            fake_tensor.data.resize_(batch_size).fill_(fake_label)

            # update D network
            #self.D_A.zero_grad()
            self.D_B.zero_grad()
            
            f_AB = self.E_AB(x_A) 
            f_BA = self.E_AB(x_B)
            #f_AB_s0=torch.zeros([f_AB.size()[0],12,f_AB.size()[2],f_AB.size()[3]])
            #f_AB_s0=Variable(f_AB_s0.cuda())
            #f_BA_s0=torch.zeros([f_BA.size()[0],12,f_BA.size()[2],f_BA.size()[3]])
            #f_BA_s0=Variable(f_BA_s0.cuda())
            f_AB_g=f_AB[:,0:500,:,:]
            f_BA_g=f_BA[:,0:500,:,:]
            f_AB_s=f_AB[:,500:512,:,:]
            f_BA_s=f_BA[:,500:512,:,:]
            f_max = torch.cat((f_AB_g, f_BA_s),1)
            #f_max = torch.max(f_AB,f_BA)
            x_AB = self.D_AB(f_max).detach()
            
            f_min = torch.cat((f_BA_g, f_AB_s),1)
            #f_min = torch.min(f_AB,f_BA)
            x_BA = self.D_AB(f_min).detach()
          
            #x_ABA = self.D_BA(torch.cat((self.E_BA(x_AB)[0], self.E_AB(x_A)[1]),1)).detach()
            #x_BAB = self.D_AB(torch.cat((self.E_AB(x_BA)[0], self.E_BA(x_B)[1]),1)).detach()

            if self.loss == "log_prob":
                l_d_A_real, l_d_A_fake = bce(self.D_B(x_A), real_tensor), bce(self.D_B(x_BA), fake_tensor)
                l_d_B_real, l_d_B_fake = bce(self.D_B(x_B), real_tensor), bce(self.D_B(x_AB), fake_tensor)
            elif self.loss == "least_square":
                l_d_A_real, l_d_A_fake = \
                    0.5 * torch.mean((self.D_B(x_A) - 1)**2), 0.5 * torch.mean((self.D_B(x_BA))**2)
                l_d_B_real, l_d_B_fake = \
                    0.5 * torch.mean((self.D_B(x_B) - 1)**2), 0.5 * torch.mean((self.D_B(x_AB))**2)
            else:
                raise Exception("[!] Unkown loss type: {}".format(self.loss))

            l_d_A = l_d_A_real + l_d_A_fake
            l_d_B = l_d_B_real + l_d_B_fake

            l_d = l_d_B + l_d_A

            l_d.backward()
            optimizer_d.step()

            # update G network
            self.D_AB.zero_grad()
            #self.D_BA.zero_grad()
            self.E_AB.zero_grad()
            #self.E_BA.zero_grad()
            
            f_AB = self.E_AB(x_A) 
            f_BA = self.E_AB(x_B)
            f_AB_g=f_AB[:,0:500,:,:]
            f_BA_g=f_BA[:,0:500,:,:]
            f_AB_s=f_AB[:,500:512,:,:]
            f_BA_s=f_BA[:,500:512,:,:]
            f_max = torch.cat((f_AB_g, f_BA_s),1)
            #f_max = torch.max(f_AB,f_BA)
            x_AB = self.D_AB(f_max)
            
            f_min = torch.cat((f_BA_g, f_AB_s),1)
            #f_min = torch.min(f_AB,f_BA)
            x_BA = self.D_AB(f_min)
            
            #f_min = torch.min(f_AB,f_BA)
            #f_min = torch.cat((f_BA, f_AB_2),1)
            #x_BA = self.D_BA(f_min)
            f_x_AB = self.E_AB(x_AB)
            f_x_AB_g=f_x_AB[:,0:500,:,:]
            f_x_AB_s=f_x_AB[:,500:512,:,:] 
            x_ABA = self.D_AB(torch.cat((f_x_AB_g, f_AB_s),1))
            
            f_x_BA = self.E_AB(x_BA)
            f_x_BA_g=f_x_BA[:,0:500,:,:]
            f_x_BA_s=f_x_BA[:,500:512,:,:] 
            x_BAB = self.D_AB(torch.cat((f_x_BA_g, f_BA_s),1))
            #x_BAB = self.D_AB(torch.cat((self.E_AB(x_BA)[0], self.E_BA(x_B)[1]),1))

            l_const_A = d(x_ABA, x_A)
            l_const_B = d(x_BAB, x_B)
            
            
            #f_x_AB, f_x_AB_2 =self.E_BA(x_AB)
            #d_x_BA =self.D_A(x_BA)
            #f_x_BA, f_x_BA_2=self.E_AB(x_BA)
            #import IPython
            #IPython.embed()
            #f_AB_g_data = Variable(f_AB_g.data, requires_grad=False)
            #l_feat_g = d(f_x_AB_g, f_AB_g_data)
            f_AB_s_data = Variable(f_AB_s.data, requires_grad=False)           
            f_BA_s_data = Variable(f_BA_s.data, requires_grad=False)
            
            l_feat_As = d(f_x_AB_s, f_BA_s_data)
            l_feat_Bs = d(f_x_BA_s, f_AB_s_data)
            
           
            if self.loss == "log_prob":
                l_gan_A = bce(self.D_B(x_BA), real_tensor)
                l_gan_B = bce(self.D_B(x_AB), real_tensor)
            elif self.loss == "least_square":
                l_gan_A = 0.5 * torch.mean((self.D_B(x_BA) - 1)**2)
                l_gan_B = 0.5 * torch.mean((self.D_B(x_AB) - 1)**2)
            else:
                raise Exception("[!] Unkown loss type: {}".format(self.loss))

            l_g =  l_gan_A + l_gan_B + l_const_A + l_const_B + 150*(l_feat_As + l_feat_Bs)
            
            l_g.backward()
            optimizer_g.step()
            '''
            # update G network with feat loss
            self.D_AB.zero_grad()
            self.D_BA.zero_grad()
            self.E_AB.zero_grad()
            self.E_BA.zero_grad()
            f_AB, gate_e = self.E_AB(x_A) 
            f_BA = self.E_BA(x_B)
            f_max = f_BA*gate_e + f_AB*(1-gate_e)
            #f_max = torch.max(f_AB,f_BA)

            x_AB = self.D_AB(f_max)
            
            #f_min = torch.min(f_AB,f_BA)
            f_min = f_AB*gate_e + f_BA*(1-gate_e)
            x_BA = self.D_BA(f_min)

            x_ABA = self.D_BA(self.E_BA(x_AB))
            x_BAB = self.D_AB(self.E_AB(x_BA)[0])
            d_x_AB, feat_AB=self.D_B(x_AB)
            d_x_B, feat_B=self.D_B(x_B)
            d_x_BA, feat_BA=self.D_A(x_BA)
            d_x_A, feat_A=self.D_A(x_A)
            l_feat_A = d(feat_AB, feat_B)
            l_feat_B = d(feat_BA, feat_A)
            l_g = l_feat_A + l_feat_B
            l_g.backward()
            optimizer_g.step()
            '''
            if step % self.log_step == 0:
                print("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}". \
                      format(step, self.max_step, l_d.data[0], l_g.data[0]))

                print("[{}/{}] l_d_A_real: {:.4f} l_d_A_fake: {:.4f}, l_d_B_real: {:.4f}, l_d_B_fake: {:.4f}". \
                      format(step, self.max_step, l_d_A_real.data[0], l_d_A_fake.data[0],
                             l_d_B_real.data[0], l_d_B_fake.data[0]))

                print("[{}/{}] l_const_A: {:.4f}, l_const_B: {:.4f}, l_gan_A: {:.4f}, l_gan_B: {:.4f}". \
                      format(step, self.max_step, l_const_A.data[0], l_const_B.data[0],
                             l_gan_A.data[0], l_gan_B.data[0]))
                
                print("[{}/{}] l_feat_As: {:.4f} l_feat_Bs: {:.4f}". \
                      format(step, self.max_step, l_feat_As.data[0], l_feat_Bs.data[0]))
                      
                #print("[{}/{}] l_feat_A_2: {:.4f} l_feat_B_2: {:.4f}". \
                #      format(step, self.max_step, l_feat_A_2.data[0], l_feat_B_2.data[0]))

                self.generate_with_A(valid_x_A, valid_x_B, self.model_dir, idx=step)
                self.generate_with_B(valid_x_A, valid_x_B, self.model_dir, idx=step)

            if step % self.save_step == self.save_step - 1:
                print("[*] Save models to {}...".format(self.model_dir))

                torch.save(self.E_AB.state_dict(), '{}/E_AB_{}.pth'.format(self.model_dir, step))
                #torch.save(self.E_BA.state_dict(), '{}/E_BA_{}.pth'.format(self.model_dir, step))
                
                torch.save(self.D_AB.state_dict(), '{}/D_AB_{}.pth'.format(self.model_dir, step))
                #torch.save(self.D_BA.state_dict(), '{}/D_BA_{}.pth'.format(self.model_dir, step))

                #torch.save(self.D_A.state_dict(), '{}/D_A_{}.pth'.format(self.model_dir, step))
                torch.save(self.D_B.state_dict(), '{}/D_B_{}.pth'.format(self.model_dir, step))

    def generate_with_A(self, inputs, inputs2, path, idx=None):
        
        f_AB = self.E_AB(inputs) 
        f_BA = self.E_AB(inputs2)
        f_AB_g=f_AB[:,0:500,:,:]
        f_BA_g=f_BA[:,0:500,:,:]
        f_AB_s=f_AB[:,500:512,:,:]
        f_BA_s=f_BA[:,500:512,:,:]
        f_max = torch.cat((f_AB_g, f_BA_s),1)
        x_AB = self.D_AB(f_max)
        
        f_x_AB = self.E_AB(x_AB)
        f_x_AB_g=f_x_AB[:,0:500,:,:]
        f_x_AB_s=f_x_AB[:,500:512,:,:] 
        x_ABA = self.D_AB(torch.cat((f_x_AB_g, f_AB_s),1))
              
        x_AB_path = '{}/{}_x_AB.png'.format(path, idx)
        x_ABA_path = '{}/{}_x_ABA.png'.format(path, idx)

        vutils.save_image(x_AB.data*0.5+0.5, x_AB_path)
        print("[*] Samples saved: {}".format(x_AB_path))
          
        vutils.save_image(x_ABA.data*0.5+0.5, x_ABA_path)
        print("[*] Samples saved: {}".format(x_ABA_path))

    def generate_with_B(self, inputs, inputs2, path, idx=None):
    
        f_AB = self.E_AB(inputs) 
        f_BA = self.E_AB(inputs2)
        f_AB_g=f_AB[:,0:500,:,:]
        f_BA_g=f_BA[:,0:500,:,:]
        f_AB_s=f_AB[:,500:512,:,:]
        f_BA_s=f_BA[:,500:512,:,:]
            
        f_min = torch.cat((f_BA_g, f_AB_s),1)
        x_BA = self.D_AB(f_min)
            
        f_x_BA = self.E_AB(x_BA)
        f_x_BA_g=f_x_BA[:,0:500,:,:]
        f_x_BA_s=f_x_BA[:,500:512,:,:] 
        x_BAB = self.D_AB(torch.cat((f_x_BA_g, f_BA_s),1))
        

        x_BA_path = '{}/{}_x_BA.png'.format(path, idx)
        x_BAB_path = '{}/{}_x_BAB.png'.format(path, idx)

        vutils.save_image(x_BA.data*0.5+0.5, x_BA_path)
        print("[*] Samples saved: {}".format(x_BA_path))

        vutils.save_image(x_BAB.data*0.5+0.5, x_BAB_path)
        print("[*] Samples saved: {}".format(x_BAB_path))

    def generate_infinitely(self, inputs, path, input_type, count=10, nrow=2, idx=None):
        if input_type.lower() == "a":
            iterator = [self.G_AB, self.G_BA] * count
        elif input_type.lower() == "b":
            iterator = [self.G_BA, self.G_AB] * count

        out = inputs
        for step, model in enumerate(iterator):
            out = model(out)

            out_path = '{}/{}_x_{}_#{}.png'.format(path, idx, input_type, step)
            vutils.save_image(out.data, out_path, nrow=nrow)
            print("[*] Samples saved: {}".format(out_path))

    def test(self):
        batch_size = self.config.sample_per_image
        A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)

        test_dir = os.path.join(self.model_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        step = 0
        #x_B = self._get_variable(B_loader.next())
        #x_B = self._get_variable(B_loader.next())
        #x_B = self._get_variable(B_loader.next())
        #x_B = self._get_variable(B_loader.next())
        #x_B = self._get_variable(B_loader.next())
        #x_B = self._get_variable(B_loader.next())
        #x_A = self._get_variable(A_loader.next())
        #x_A = self._get_variable(A_loader.next())
        #x_A = self._get_variable(A_loader.next())
        #x_A = self._get_variable(A_loader.next())
        while True:
            try:
                x_A,x_B = self._get_variable(A_loader.next()), self._get_variable(B_loader.next())
            except StopIteration:
                print("[!] Test sample generation finished. Samples are in {}".format(test_dir))
                break

            vutils.save_image(x_A.data*0.5+0.5, '{}/{}_x_A.png'.format(test_dir, step))
            vutils.save_image(x_B.data*0.5+0.5, '{}/{}_x_B.png'.format(test_dir, step))
            #print(x_A.size())
            #print(x_B.size())
            self.generate_with_A(x_A, x_B, test_dir, idx=step)
            self.generate_with_B(x_A, x_B, test_dir, idx=step)

            #self.generate_infinitely(x_A, test_dir, input_type="A", count=10, nrow=4, idx=step)
            #self.generate_infinitely(x_B, test_dir, input_type="B", count=10, nrow=4, idx=step)

            step += 1

    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
