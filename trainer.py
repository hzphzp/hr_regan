from __future__ import print_function
import numpy as np
import os
from glob import glob
# from tqdm import trange
from itertools import chain

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

from models import *
from data_loader import get_loader
from img_random_discmp import img_random_dis


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
        self.dataset= config.dataset

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
            self.D_H.cuda()
            self.D_L.cuda()
            self.D_F.cuda()
            self.D_AB.cuda()
            self.D_BA.cuda()
            self.D_FB.cuda()
            self.E_AB.cuda()

        elif self.num_gpu > 1:
            self.D_H = nn.DataParallel(self.D_H.cuda(),device_ids=range(self.num_gpu))
            self.D_L = nn.DataParallel(self.D_L.cuda(),device_ids=range(self.num_gpu))
            self.D_F = nn.DataParallel(self.D_F.cuda(),device_ids=range(self.num_gpu))
            self.D_AB = nn.DataParallel(self.D_AB.cuda(),device_ids=range(self.num_gpu))
            self.D_BA = nn.DataParallel(self.D_BA.cuda(),device_ids=range(self.num_gpu))
            self.D_FB = nn.DataParallel(self.D_FB.cuda(),device_ids=range(self.num_gpu))
            self.E_AB = nn.DataParallel(self.E_AB.cuda(),device_ids=range(self.num_gpu))

        if self.load_path:
            self.load_model()
    @staticmethod
    def psnr(original, compared):
        d = nn.MSELoss()
        arg_psnr = 0
        for i in range(original.size(0)):
            mse = d(original[i], compared[i])
            try:
                psnr = 10 * torch.log(4/ mse)/np.log(10)
            except:
                pass
            arg_psnr = arg_psnr + psnr
        arg_psnr = arg_psnr/original.size(0)
        return arg_psnr

    def build_model(self):
        if self.dataset == 'toy':
            self.D_H = DiscriminatorFC(2, 1, [self.config.fc_hidden_dim] * self.config.d_num_layer)
            self.D_L = DiscriminatorFC(2, 1, [self.config.fc_hidden_dim] * self.config.d_num_layer)
        else:
            a_height, a_width, a_channel = self.a_data_loader.shape
            b_height, b_width, b_channel = self.b_data_loader.shape
            a_channel = 1
            b_channel = 1 
            if self.cnn_type == 0:
                #conv_dims, deconv_dims = [64, 128, 256, 512], [512, 256, 128, 64]
                conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
            elif self.cnn_type == 1:
                #conv_dims, deconv_dims = [32, 64, 128, 256], [256, 128, 64, 32]
                conv_dims, deconv_dims = [32, 64, 128, 256], [128, 64, 32]
            else:
                raise Exception("[!] cnn_type {} is not defined".format(self.cnn_type))

            self.D_H = DiscriminatorCNN(
                    a_channel, 1, conv_dims, self.num_gpu)
            self.D_L = DiscriminatorCNN(
                    b_channel, 1, conv_dims, self.num_gpu)
            self.D_F = DiscriminatorCNN(
                    a_channel, 1, conv_dims, self.num_gpu)
            self.D_AB = DecoderCNN(
                    conv_dims[-1], b_channel, deconv_dims, self.num_gpu)
            self.D_BA = DecoderCNN(
                    conv_dims[-1], b_channel, deconv_dims, self.num_gpu)
            self.D_FB = DecoderCNN(
                    conv_dims[-1], b_channel, deconv_dims, self.num_gpu)
            self.E_AB = EncoderCNN_1(
                    a_channel, conv_dims, self.num_gpu)

            self.D_H.apply(weights_init)
            self.D_L.apply(weights_init)
            self.D_F.apply(weights_init)
            self.D_AB.apply(weights_init)
            self.D_BA.apply(weights_init)
            self.D_FB.apply(weights_init)
            self.E_AB.apply(weights_init)

    def load_model(self):
        #TODO:how to load model
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_AB_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [int(os.path.basename(path.split('.')[0].split('_')[-1])) for path in paths]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        G_AB_filename = '{}/G_AB_{}.pth'.format(self.load_path, self.start_step)
        #todo: to change the load model function
        '''
        self.G_AB.load_state_dict(torch.load(G_AB_filename, map_location=map_location))
        self.G_BA.load_state_dict(
            torch.load('{}/G_BA_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

        self.D_A.load_state_dict(
            torch.load('{}/D_A_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))
        self.D_B.load_state_dict(
            torch.load('{}/D_B_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))
        '''
        print("[*] Model loaded: {}".format(G_AB_filename))

    def train(self):
        d = nn.MSELoss()
        bce = nn.BCELoss()

        real_label = 1
        fake_label = 0

        real_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = real_tensor.data.fill_(real_label)

        fake_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = fake_tensor.data.fill_(fake_label)

        rlfk_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = rlfk_tensor.data.fill_(0.5)

        if self.num_gpu > 0:
            d.cuda()
            bce.cuda()

            real_tensor = real_tensor.cuda()
            fake_tensor = fake_tensor.cuda()
            rlfk_tensor = rlfk_tensor.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(self.config.optimizer))

        optimizer_1_g = optimizer(
            chain(self.E_AB.parameters(), self.D_AB.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_1_d = optimizer(
            chain(self.D_H.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_2_g = optimizer(
            chain(self.E_AB.parameters(), self.D_BA.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_2_d = optimizer(
            chain(self.D_L.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2))
        optimizer_3_g = optimizer(
            chain(self.E_AB.parameters(), self.D_FA.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_3_d = optimizer(
            chain(self.D_F.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2))

        A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
        valid_x_A, valid_x_B = A_loader.next(), B_loader.next()
        x_A_t2a=valid_x_A.numpy()
        #x_B_t2a=x_B.numpy()
        x_A_t2a, x_B_t2a =img_random_dis(x_A_t2a)
        valid_x_A=torch.from_numpy(x_A_t2a)
        valid_x_B=torch.from_numpy(x_B_t2a)
        valid_x_A=valid_x_A.float()
        valid_x_B=valid_x_B.float()
        valid_x_A, valid_x_B = self._get_variable(valid_x_A), self._get_variable(valid_x_B)

        vutils.save_image(valid_x_A.data, '{}/valid_x_A.png'.format(self.model_dir))
        vutils.save_image(valid_x_B.data, '{}/valid_x_B.png'.format(self.model_dir))

        for step in range(self.start_step, self.max_step):
            try:
                x_A_1, x_B_1 = A_loader.next(), B_loader.next()
            except StopIteration:
                A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
                x_A_1, x_B_1 = A_loader.next(), B_loader.next()
            if x_A_1.size(0) != x_B_1.size(0):
                print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                continue

            x_A_t2a=x_A_1.numpy()
            x_A_t2a, x_B_t2a =img_random_dis(x_A_t2a)
            
            x_A=torch.from_numpy(x_A_t2a)
            x_B=torch.from_numpy(x_B_t2a)
            x_A=x_A.float()
            x_B=x_B.float()
            x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)

            batch_size = x_A.size(0)
            real_tensor.data.resize_(batch_size).fill_(real_label)
            fake_tensor.data.resize_(batch_size).fill_(fake_label)
            rlfk_tensor.data.resize_(batch_size).fill_(0.5)

            '''update the first model: L to H'''
            # update D_H network
            self.D_H.zero_grad()

            f_AB = self.E_AB(x_A)
            f_AB_g = f_AB[:, 0:511:, :, :]
            f_AB_s = f_AB[:, 511:512, :, :]
            f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
            f_AB_s0 = Variable(f_AB_s0.cuda())
            f_AB_g0 = torch.zeros([f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]])
            f_AB_g0 = Variable(f_AB_g0.cuda())

            f_max = torch.cat((f_AB_g, f_AB_s0), 1)
            f_min = torch.cat((f_AB_g0, f_AB_s), 1)
            x_H = self.D_AB(f_max).detach()

            if self.loss == "log_prob":
                l_dh_B_real, l_dh_B_fake = bce(self.D_H(x_B), real_tensor), bce(self.D_H(x_H), fake_tensor)
            elif self.loss == "least_square":
                l_dh_B_real, l_dh_B_fake = \
                    0.5 * torch.mean((self.D_H(x_B) - 1)**2), 0.5 * torch.mean((self.D_H(x_H))**2)
            else:
                raise Exception("[!] Unknown loss type: {}".format(self.loss))

            l_dh_B = l_dh_B_real + l_dh_B_fake

            l_dh = l_dh_B

            l_dh.backward()
            optimizer_1_d.step()

            # update D_AB network
            for gab_step in range(100):
                try:
                    x_A_1, x_B_1 = A_loader.next(), B_loader.next()
                except StopIteration:
                    A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
                    x_A_1, x_B_1 = A_loader.next(), B_loader.next()
                if x_A_1.size(0) != x_B_1.size(0):
                    print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                    continue
                
                x_A_t2a=x_A_1.numpy()
                x_A_t2a, x_B_t2a =img_random_dis(x_A_t2a)
                
                x_A=torch.from_numpy(x_A_t2a)
                x_B=torch.from_numpy(x_B_t2a)
                x_A=x_A.float()
                x_B=x_B.float()
                x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)

                batch_size = x_A.size(0)
                real_tensor.data.resize_(batch_size).fill_(real_label)
                fake_tensor.data.resize_(batch_size).fill_(fake_label)

                self.E_AB.zero_grad()
                self.D_AB.zero_grad()

                f_AB = self.E_AB(x_A)
                f_AB_g = f_AB[:, 0:511:, :, :]
                f_AB_s = f_AB[:, 511:512, :, :]
                f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
                f_AB_s0 = Variable(f_AB_s0.cuda())
                f_AB_g0 = torch.zeros([f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]])
                f_AB_g0 = Variable(f_AB_g0.cuda())

                f_max = torch.cat((f_AB_g, f_AB_s0), 1)
                f_min = torch.cat((f_AB_g0, f_AB_s), 1)
                x_H = self.D_AB(f_max)
                
                l_const_AB = d(x_H, x_B)
                
                dh_x_AB = self.D_H(x_H)

                if self.loss == "log_prob":
                    l_gan_AB = bce(dh_x_AB, real_tensor)
                elif self.loss == "least_square":
                    l_gan_AB = 0.5 * torch.mean((dh_x_AB - 1)**2)
                else:
                    raise Exception("[!] Unkown loss type: {}".format(self.loss))

                l_gh = l_const_AB + l_gan_AB

                l_gh.backward()
                optimizer_1_g.step()

            '''update the second model: L to L'''
            # update D_L network
            self.D_L.zero_grad()

            f_AB = self.E_AB(x_A)
            f_AB_g = f_AB[:, 0:511:, :, :]
            f_AB_s = f_AB[:, 511:512, :, :]
            f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
            f_AB_s0 = Variable(f_AB_s0.cuda())
            f_AB_g0 = torch.zeros([f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]])
            f_AB_g0 = Variable(f_AB_g0.cuda())

            f_max = torch.cat((f_AB_g, f_AB_s0), 1)
            f_min = torch.cat((f_AB_g0, f_AB_s), 1)
            x_L = self.D_BA(f_AB).detach()

            if self.loss == "log_prob":
                l_dl_A_real, l_dl_A_fake = bce(self.D_L(x_A), real_tensor), bce(self.D_L(x_L), fake_tensor)
            elif self.loss == "least_square":
                l_dl_A_real, l_dl_A_fake = \
                    0.5 * torch.mean((self.D_L(x_A) - 1) ** 2), 0.5 * torch.mean((self.D_L(x_L)) ** 2)
            else:
                raise Exception("[!] Unknown loss type: {}".format(self.loss))

            l_dl_A = l_dl_A_real + l_dl_A_fake

            l_dl = l_dl_A

            l_dl.backward()
            optimizer_2_d.step()

            # update D_AB network
            for gba_step in range(100):
                try:
                    x_A_1, x_B_1 = A_loader.next(), B_loader.next()
                except StopIteration:
                    A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
                    x_A_1, x_B_1 = A_loader.next(), B_loader.next()
                if x_A_1.size(0) != x_B_1.size(0):
                    print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                    continue

                x_A_t2a = x_A_1.numpy()
                x_A_t2a, x_B_t2a = img_random_dis(x_A_t2a)

                x_A = torch.from_numpy(x_A_t2a)
                x_B = torch.from_numpy(x_B_t2a)
                x_A = x_A.float()
                x_B = x_B.float()
                x_A, x_B = self._get_variable(x_A), self._get_variable(x_B)

                batch_size = x_A.size(0)
                real_tensor.data.resize_(batch_size).fill_(real_label)
                fake_tensor.data.resize_(batch_size).fill_(fake_label)

                self.E_AB.zero_grad()
                self.D_BA.zero_grad()

                f_AB = self.E_AB(x_A)
                f_AB_g = f_AB[:, 0:511:, :, :]
                f_AB_s = f_AB[:, 511:512, :, :]
                f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
                f_AB_s0 = Variable(f_AB_s0.cuda())
                f_AB_g0 = torch.zeros([f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]])
                f_AB_g0 = Variable(f_AB_g0.cuda())

                f_max = torch.cat((f_AB_g, f_AB_s0), 1)
                f_min = torch.cat((f_AB_g0, f_AB_s), 1)

                x_L = self.D_BA(f_AB)

                l_const_AFA = d(x_L, x_A)

                dl_x_AA = self.D_L(x_L)

                if self.loss == "log_prob":
                    l_gan_AA = bce(dl_x_AA, real_tensor)
                elif self.loss == "least_square":
                    l_gan_AA = 0.5 * torch.mean((dl_x_AA - 1) ** 2)
                else:
                    raise Exception("[!] Unkown loss type: {}".format(self.loss))

                l_gl = l_const_AFA + l_gan_AA

                l_gl.backward()
                optimizer_2_g.step()

            '''update the third model, F to ?'''
            self.D_F.zero_grad()

            f_AB = self.E_AB(x_A)
            f_AB_g = f_AB[:, 0:511:, :, :]
            f_AB_s = f_AB[:, 511:512, :, :]
            f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
            f_AB_s0 = Variable(f_AB_s0.cuda())
            f_AB_g0 = torch.zeros([f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]])
            f_AB_g0 = Variable(f_AB_g0.cuda())

            f_max = torch.cat((f_AB_g, f_AB_s0), 1)
            f_min = torch.cat((f_AB_g0, f_AB_s), 1)

            x_FB = self.D_FB(f_AB).detach()

            x_max = self.D_FB(f_max).detach()

            if self.loss == "log_prob":
                l_df_B_real, l_df_B_fake = bce(self.D_F(x_FB), rlfk_tensor + 0.1), bce(self.D_F(x_max),
                                                                                     rlfk_tensor - 0.1)
            elif self.loss == "least_square":
                l_df_B_real, l_df_B_fake = \
                    0.5 * torch.mean((self.D_F(x_FB) - 0.6) ** 2), 0.5 * torch.mean((self.D_F(x_max) - 0.4) ** 2)
            else:
                raise Exception("[!] Unkown loss type: {}".format(self.loss))

            l_df_B = l_df_B_real + l_df_B_fake

            l_df = l_df_B

            l_df.backward()
            optimizer_3_d.step()

            # update G_FB network
            for gfb_step in range(2):
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
                rlfk_tensor.data.resize_(batch_size).fill_(0.5)

                self.D_FB.zero_grad()
                self.E_AB.zero_grad()

                f_AB = self.E_AB(x_A)
                f_AB_g = f_AB[:, 0:511:, :, :]
                f_AB_s = f_AB[:, 511:512, :, :]
                f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
                f_AB_s0 = Variable(f_AB_s0.cuda())
                f_AB_g0 = torch.zeros([f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]])
                f_AB_g0 = Variable(f_AB_g0.cuda())

                f_max = torch.cat((f_AB_g, f_AB_s0), 1)
                f_min = torch.cat((f_AB_g0, f_AB_s), 1)

                x_FB = self.D_FB(f_AB)
                x_min = self.D_FB(f_min)
                x_max = self.D_FB(f_max)

                l_const_FA = d(x_FB, x_A)

                l_const_min = d(x_min, x_A)

                l_const_max = d(x_max, x_A)

                if self.loss == "log_prob":
                    l_gan_A = bce(self.D_F(x_FB), rlfk_tensor + 0.1)
                    l_gan_B = bce(self.D_F(x_max), rlfk_tensor - 0.1)
                elif self.loss == "least_square":
                    l_gan_A = 0.5 * torch.mean((self.D_F(x_FB) - 0.6) ** 2)
                    l_gan_B = 0.5 * torch.mean((self.D_F(x_max) - 0.4) ** 2)
                else:
                    raise Exception("[!] Unkown loss type: {}".format(self.loss))

                l_gf = 1 * (l_const_FA) + l_gan_A + l_gan_B
                l_gf.backward()
                optimizer_3_g.step()

            if step % self.log_step == 0:
                print("[{}/{}] l_dh: {:.4f} l_gan_AB: {:.4f} l_const_AB: {:.4f}". \
                      format(step, self.max_step, l_dh.data[0], l_dl.data[0], l_const_AB.data[0]))

                print("[{}/{}] l_dl: {:.4f} l_gan_AA: {:.4f}, l_const_AFA: {:.4f}". \
                      format(step, self.max_step, l_dl.data[0], l_gan_AA.data[0],
                             l_const_AFA.data[0] ))

                print("[{}/{}] l_df: {:.4f} l_gan_A: {:.4f} l_gan_B: {:.4f} l_const_FA: {:.4f}". \
                      format(step, self.max_step, l_df.data[0], l_gan_A.data[0], l_gan_B.data[0],
                             l_const_FA.data[0] ))

                self.generate_with_A(valid_x_A, self.model_dir, idx=step)
                # self.generate_with_B(valid_x_B, self.model_dir, idx=step)

            if step % self.save_step == self.save_step - 1:
                print("[*] Save models to {}...".format(self.model_dir))

                torch.save(self.E_AB.state_dict(), '{}/E_AB_{}.pth'.format(self.model_dir, step))
                torch.save(self.D_AB.state_dict(), '{}/D_AB_{}.pth'.format(self.model_dir, step))

                # torch.save(self.D_A.state_dict(), '{}/D_A_{}.pth'.format(self.model_dir, step))
                # torch.save(self.D_B.state_dict(), '{}/D_B_{}.pth'.format(self.model_dir, step))

    def generate_with_A(self, inputs, path, idx=None):
        f_AB = self.E_AB(inputs)
        f_AB_g = f_AB[:, 0:511:, :, :]
        f_AB_s = f_AB[:, 511:512, :, :]
        f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
        f_AB_s0 = Variable(f_AB_s0.cuda())
        f_AB_g0 = torch.zeros([f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]])
        f_AB_g0 = Variable(f_AB_g0.cuda())

        f_max = torch.cat((f_AB_g, f_AB_s0), 1)
        f_min = torch.cat((f_AB_g0, f_AB_s), 1)
        x_H = self.D_AB(f_max)
        x_S = self.D_AB(f_min)

        x_H_path = '{}/{}_x_H.png'.format(path, idx)
        x_S_path = '{}/{}_x_L.png'.format(path, idx)

        vutils.save_image(x_H.data, x_H_path)
        print("[*] Samples saved: {}".format(x_H_path))

        vutils.save_image(x_S.data, x_S_path)
        print("[*] Samples saved: {}".format(x_S_path))

    def generate_with_A_test(self, inputs, inputs2, path, idx=None):

        f_AB = self.E_AB(inputs)

        f_AB_s0 = torch.zeros([f_AB.size()[0], 1, f_AB.size()[2], f_AB.size()[3]])
        f_AB_s0 = Variable(f_AB_s0)
        f_AB_s = f_AB[:, 511:512, :, :]

        f_AB_g0 = torch.Tensor(f_AB.size()[0], 511, f_AB.size()[2], f_AB.size()[3]).uniform_(0,
                                                                                             1)  # torch.zeros([f_AB.size()[0],511,f_AB.size()[2],f_AB.size()[3]])
        print(f_AB_g0.size())
        f_AB_g0 = Variable(f_AB_g0)
        f_AB_g = f_AB[:, 0:511, :, :]

        f_max = torch.cat((f_AB_g0, f_AB_s), 1)
        x_AB = self.D_AB(f_AB)
        x_max = self.D_AB(f_max)

        x_AB_path = '{}/{}_x_AB.png'.format(path, idx)
        x_max_path = '{}/{}_x_max.png'.format(path, idx)

        vutils.save_image(x_AB.data, x_AB_path)
        print("[*] Samples saved: {}".format(x_AB_path))

        vutils.save_image(x_max.data, x_max_path)
        print("[*] Samples saved: {}".format(x_max_path))

    '''
    def generate_with_B(self, inputs, path, idx=None):
        x_BA = self.G_BA(inputs)
        x_BAB = self.G_AB(x_BA)

        x_BA_path = '{}/{}_x_BA.png'.format(path, idx)
        x_BAB_path = '{}/{}_x_BAB.png'.format(path, idx)

        vutils.save_image(x_BA.data, x_BA_path)
        print("[*] Samples saved: {}".format(x_BA_path))

        vutils.save_image(x_BAB.data, x_BAB_path)
        print("[*] Samples saved: {}".format(x_BAB_path))
    '''

    def generate_infinitely(self, inputs, path, input_type, count=10, nrow=2, idx=None):
        # todo to debug
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
        while True:
            try:
                x_A,x_B = self._get_variable(A_loader.next()), self._get_variable(B_loader.next())
            except StopIteration:
                print("[!] Test sample generation finished. Samples are in {}".format(test_dir))
                break

            vutils.save_image(x_A.data, '{}/{}_x_A.png'.format(test_dir, step))
            vutils.save_image(x_B.data, '{}/{}_x_B.png'.format(test_dir, step))

            self.generate_with_A_test(x_A, x_B, test_dir, idx=step)
            step += 1

    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
