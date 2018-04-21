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
        self.FPS = 3 #5 FRAMES PER SECOND
        self.image = None
        self.quit = False
        self.paused = True
        self.action = 0
        self.treads = [0,0]
        self.speed=.5
        self.timeStart = time.time()
        self.lr = 0.5
        self.imsz = [240//3, 320//3]
        self.ps = 21
        self.D = torch.randn(3*self.ps**2, 100).float().cuda(0)
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
        #ZCAMatrix = torch.mm(U, torch.mm(torch.diag(1.0/torch.sqrt(S + epsilon)), torch.t(U)))

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




    def X3(self, x, D):
        e = 1e-8
        
        x = torch.from_numpy(x).float().cuda(0)
        x = x.unfold(0, self.ps, 1).unfold(1, self.ps, 1).unfold(2, 3, 1)
        x = x.contiguous().view(x.size(0)*x.size(1)*x.size(2),
                                x.size(3)*x.size(4), x.size(-1))
        x = x - torch.mean(x, 0)
        x = torch.t(x.view(-1, x.size(1)*3))
        x = self.whiten(x)
        
        D = torch.mm(D, torch.diag(1./(torch.sqrt(torch.sum(D**2, 0))+e)))
        a = torch.mm(torch.t(D), x).cuda(0)
        a = torch.mm(a, torch.diag(1./(torch.sqrt(torch.sum(a**2, 0))+e)))
        a = .3 * a ** 3
        x = x - torch.mm(D, a)
        D = D + torch.mm(x, torch.t(a))

        return D, a



    def process_video_from_rover(self, jpegbytes, timestamp_10msec):
        array_of_bytes = np.fromstring(jpegbytes, np.uint8)
        self.image = cv2.imdecode(array_of_bytes, flags=3)
        k = cv2.waitKey(1) & 0xFF
        return self.image


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
            self.D, self.a = self.X3(self.image, self.D)
            cv2.namedWindow('dictionary', cv2.WINDOW_NORMAL)
            cv2.imshow('dictionary', self.montage(self.mat2ten(self.D.cpu().numpy())))
            cv2.waitKey(1)
            
            self.clock.tick(self.FPS)
            pygame.display.flip()
            self.move_camera_in_vertical_direction(0)


        pygame.quit()
        cv2.destroyAllWindows()
        self.close()

