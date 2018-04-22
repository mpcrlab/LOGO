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
from skimage.util import view_as_windows as vaw
from torchvision.transforms import Pad

class RoverBrain(Rover):
    def __init__(self):
        Rover.__init__(self)
        self.userInterface = Pygame_UI()
        self.clock = pygame.time.Clock()
        self.FPS = 20  # FRAMES PER SECOND
        self.image = None  # incoming image
        self.quit = False  
        self.paused = True
        self.action = 0  # what action to do
        self.treads = [0,0]  # steering/throttle action
	self.count = 0  
        self.speed=.5  # change the vehicle's speed here
        self.ts = time.time()
        self.lr = 0.2 # learning rate
        self.imsz = [240//3, 320//3] # size to reshape self.image to
        self.prune = 1
        self.ps = 15
        self.k = 81
        self.k2 = 81
        self.D = torch.randn(3*self.ps**2, self.k).float().cuda(0)
        self.D_2 = torch.randn(3*self.ps**2, self.k2).float().cuda(0)
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




    def X3(self, x, D, D_2):
        e = 1e-8  # constant to avoid div. by 0.
        
        # prepare x, normalize, whiten, etc.
        x = torch.from_numpy(x).float().cuda(0)
        x = x.unfold(0, self.ps, 1).unfold(1, self.ps, 1).unfold(2, 3, 1)
        x = x.contiguous().view(x.size(0)*x.size(1)*x.size(2),
                                x.size(3)*x.size(4), x.size(-1))
        x = x - torch.mean(x, 0)
        x = torch.t(x.view(-1, x.size(1)*3))
        x = self.whiten(x)
        
        # scale each patch between 0 and 1
        D = torch.mm(D, torch.diag(1./(torch.sqrt(torch.sum(D**2, 0))+e)))
        
        # see how much each neuron fires for each patch
        a = torch.mm(torch.t(D), x).cuda(0)

        # lateral inhibition...scaling coefficients to between 0 and 1
        a = torch.mm(a, torch.diag(1./(torch.sqrt(torch.sum(a**2, 0))+e)))

        # cubic activation function
        a = (self.lr - self.count/1000) * a ** 3

        # update dictionary based on Hebbian learning rule
        D = D + torch.mm(x - torch.mm(D, a), torch.t(a))

        ############ second round --- hierarchical features #################

        D_2 = torch.mm(D_2, 
                       torch.diag(1./(torch.sqrt(torch.sum(D_2**2, 0))+e)))
        a_2 = torch.mm(torch.t(D_2), 
                       self.whiten(D - torch.mean(D, 1)[:, None]))
        a_2 = torch.mm(a_2, 
                       torch.diag(1./(torch.sqrt(torch.sum(a_2**2, 0))+e)))
        a_2 = (self.lr*2 - self.count/1000) * a_2 ** 3
        D_2 = D_2 + torch.mm(D - torch.mm(D_2, a_2), torch.t(a_2))

        return D, D_2


#############################################################################
    def run(self):

        while type(self.image) == type(None): 
            pass

        while not self.quit:
            #self.displayDashboard()

       	    key = self.getActiveKey()

            if key:
                key = chr(key)

                if key in ['w','a','d','s','q', 'z', 'i', 'm']:
		    if key == 'w':
		        self.action = 0
                        self.set_wheel_treads(self.speed,self.speed)

		    elif key == 'a':
		        self.action = -1
                        self.set_wheel_treads(-self.speed,self.speed)

		    elif key == 'd':
		        self.action = 1
                        self.set_wheel_treads(self.speed,-self.speed)

		    elif key == 's':
		        self.action = 2
                        self.set_wheel_treads(-self.speed,-self.speed)

		    elif key == 'q':
		        self.action = 9
		        self.set_wheel_treads(0,0)

                    elif key == 'z':
                        self.set_wheel_treads(0,0)
                        self.quit = True

                    elif key == 'i':
                        self.move_camera_in_vertical_direction(1)

                    elif key == 'm':
                        self.move_camera_in_vertical_direction(-1)


            self.image = imresize(self.image, self.imsz)
            self.D, self.D_2 = self.X3(self.image, self.D, self.D_2)
		
	    if self.count % (self.FPS*2) == 0 or self.count == 0:
            	cv2.namedWindow('dictionary', cv2.WINDOW_NORMAL)
            	cv2.imshow('dictionary', 
                           self.montage(self.mat2ten(self.D_2.cpu().numpy())))
            	cv2.waitKey(1)  
           
            if self.count % (self.FPS * 20) == 0:
                rk = np.random.randint(0, self.D.size(1), 1)[0]  
                rk_2 = np.random.randint(0, self.D_2.size(1), 1)[0]
                self.D[:, rk] = torch.randn(self.D.size(0),)  
                self.D_2[:, rk_2] = torch.randn(self.D_2.size(0),)    

            self.clock.tick(self.FPS)
            pygame.display.flip()
            self.count += 1

            if key in ['i', 'm']:
                self.move_camera_in_vertical_direction(0)

        pygame.quit()
        cv2.destroyAllWindows()
        self.close()
