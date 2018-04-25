from __future__ import print_function
import pygame
from Pygame_UI import *
from rover import Rover
import cv2
import numpy as np
import time
import math
from scipy.misc import bytescale, imresize
import torch
import torch.nn.functional as f
import sys, os


class RoverBrain(Rover):
    def __init__(self, driver):
        Rover.__init__(self)
        self.userInterface = Pygame_UI()
        self.clock = pygame.time.Clock()
        self.FPS = 10  # FRAMES PER SECOND
        self.image = None  # incoming image
        self.quit = False
        self.driver = driver
        self.action = 0  # what action to do
        self.count = 0
        self.speed = .5  # change the vehicle's speed here
        self.const = 0.3
        self.lr = 2.
        self.downsample = 2
        self.imsz = np.asarray([240//3, 320//3])
        self.action_dict = {}
        self.cam_dict = {}
        self.action_dict['w'] = [self.speed, self.speed]
        self.action_dict['a'] = [-self.speed, self.speed]
        self.action_dict['s'] = [-self.speed, -self.speed]
        self.action_dict['d'] = [self.speed, -self.speed]
        self.action_dict['q'] = [0, 0]
        self.cam_dict['i'] = 1
        self.cam_dict['m'] = -1
        self.state_act = [97, 105, 100, 97, 119, 100, 97, 109, 100]
        self.ps = 15
        self.k = 700
        self.k2 = 700
        self.D = torch.randn(3*self.ps**2, self.k).float().cuda(0)
        self.num_rows, self.num_cols = self.imsz - self.ps
        self.a_2 = torch.zeros(self.k2, self.num_rows*self.num_cols).cuda(0)
        self.D_2 = torch.randn(self.k, self.k2).float().cuda(0)
        self.run()


    def getActiveKey(self):
        key = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = event.key
        return key



    def whiten(self, X):
        '''Function to ZCA whiten image matrix.'''
        U,S,V = torch.svd(torch.mm(X, torch.t(X)))
        epsilon = 1e-5
        ZCAMatrix = torch.diag(1.0/torch.sqrt(S + epsilon))
        ZCAMatrix = torch.mm(U, torch.mm(ZCAMatrix, torch.t(U)))

        return torch.mm(ZCAMatrix, X)


    def mat2ten(self, X, c=3):
        zs=[X.shape[1], int(np.sqrt(X.shape[0]//c)), int(np.sqrt(X.shape[0]//c)), c]
        Z=np.zeros(zs)

        for i in range(X.shape[1]):
            Z[i, ...] = X[:,i].reshape([zs[1],zs[2], c])

        return Z



    def montage(self, X):
        count, m, n, c = np.shape(X)
        mm = int(np.ceil(np.sqrt(count)))
        nn = mm
        M = np.zeros((mm * m, nn * n, c))

        image_id = 0
        for j in range(mm):
            for k in range(nn):
                if image_id >= count:
                    break
                sliceM, sliceN = j * m, k * n
                M[sliceM:sliceM + m, sliceN:sliceN + n, :] = bytescale(X[image_id, ...])
                image_id += 1

        return np.uint8(M)


    def salience(self, x):
        horiz = self.num_rows // 3
        vert = self.num_cols // 3

        s = np.zeros([9,])
        s_name = ['upper left',
                  'upper center',
                  'upper right',
                  'center left',
                  'center center',
                  'center right',
                  'lower left',
                  'lower center',
                  'lower right']

        x = torch.abs(x)
        u_l = x.unfold(1, vert, vert*3)[:, :horiz, ...]
        u_c = x[vert:vert*2, :].unfold(1, vert, vert*3)[:, :horiz, ...]
        u_r = x[vert*2:, :].unfold(1, vert, vert*3)[:, :horiz, ...]
        c_l = x.unfold(1, vert, vert*3)[:, horiz:horiz*2, ...]
        c_c = x[vert:vert*2, :].unfold(1, vert, vert*3)[:, horiz:horiz*2, ...]
        c_r = x[vert*2:, :].unfold(1, vert, vert*3)[:, horiz:horiz*2]
        l_l = x.unfold(1, vert, vert*3)[:, horiz*2:, ...]
        l_c = x[vert:vert*2, :].unfold(1, vert, vert*3)[:, horiz*2:, ...]
        l_r = x[vert*2:, :].unfold(1, vert, vert*3)[:, horiz*2:, ...]

        s[0] = torch.mean(u_l)
        s[1] = torch.mean(u_c)
        s[2] = torch.mean(u_r)
        s[3] = torch.mean(c_l)
        s[4] = torch.mean(c_c)
        s[5] = torch.mean(c_r)
        s[6] = torch.mean(l_l)
        s[7] = torch.mean(l_c)
        s[8] = torch.mean(l_r)
        return self.state_act[np.argmax(s)]



    def X3(self, x, D, D_2):
        e = 1e-8  # constant to avoid div. by 0.

        # prepare x, normalize, whiten, etc.
        x = (torch.from_numpy(x).float().cuda(0)).unfold(0,
             self.ps, 1).unfold(1, self.ps, 1).unfold(2, 3, 1)
        x = x.contiguous().view(x.size(0)*x.size(1)*x.size(2),
                                x.size(3)*x.size(4), x.size(-1))
        x = x - torch.mean(x, 0)
        x = self.whiten(torch.t(x.view(-1, x.size(1)*3)))
        #x = self.whiten(x)

        # scale each patch between 0 and 1
        D = torch.mm(D, torch.diag(1./(torch.sqrt(torch.sum(D**2, 0))+e)))

        # see how much each neuron fires for each patch
        a = torch.mm(torch.t(D), x).cuda(0)

        # lateral inhibition...scaling coefficients to between 0 and 1
        a = torch.mm(a, torch.diag(1./(torch.sqrt(torch.sum(a**2, 0))+e)))

        # cubic activation function
        a = self.const * a ** 3

        #x = x - torch.mm(D, a)

        # update dictionary based on Hebbian learning rule
        D = D + self.lr * torch.mm(x - torch.mm(D, a), torch.t(a))

        ############ second round --- abstract features #################

        D_2 = torch.mm(D_2,
                       torch.diag(1./(torch.sqrt(torch.sum(D_2**2, 0))+e)))
        a = self.whiten(a - torch.mean(a, 1)[:, None])
        a_2 = torch.mm(torch.t(D_2), a)
        a_2 = torch.mm(a_2,
                       torch.diag(1./(torch.sqrt(torch.sum(a_2**2, 0))+e)))
        a_2 = self.const * a_2 ** 3
        D_2 = D_2 + self.lr * torch.mm(a - torch.mm(D_2, a_2), torch.t(a_2))

        return D, D_2, a - torch.mm(D_2, a_2)


#############################################################################
    def run(self):
        while type(self.image) == type(None):
            pass

        while not self.quit:
            self.image = imresize(self.image, self.imsz)
            self.D, self.D_2, self.a_2 = self.X3(self.image,
                                                 self.D,
                                                 self.D_2)

            # get the key the user pressed if they pressed one
            key = self.getActiveKey()
            if not key and self.driver in ['AI', 'ai', 'ml']:
                key = self.salience(self.a_2)

            if key:
                self.action = chr(key)

                if self.action in self.action_dict:
                    act = self.action_dict[self.action]
                    self.set_wheel_treads(act[0], act[1])

                elif self.action in self.cam_dict:
                    act = self.cam_dict[self.action]
                    self.move_camera_in_vertical_direction(act)

                elif self.action == 'z':
                    self.set_wheel_treads(0,0)
                    self.quit = True


            if self.count % (self.FPS*2) == 0 or self.count == 0:
            	cv2.namedWindow('dictionary', cv2.WINDOW_NORMAL)
            	cv2.imshow('dictionary', #self.image)
                           self.montage(self.mat2ten(
                           self.D.cpu().numpy())))
            	cv2.waitKey(1)
            elif self.count % (self.FPS * 15) == 0:
                rk = np.random.randint(0, self.D.size(1), 1)[0]
                rk_2 = np.random.randint(0, self.D_2.size(1), 1)[0]
                self.D[:, rk] = torch.randn(self.D.size(0),)
                self.D_2[:, rk_2] = torch.randn(self.D_2.size(0),)


            self.clock.tick(self.FPS)
            pygame.display.flip()
            self.count += 1

            if self.action in self.cam_dict:
                time.sleep(0.2)
                self.move_camera_in_vertical_direction(0)


        pygame.quit()
        cv2.destroyAllWindows()
        self.close()
