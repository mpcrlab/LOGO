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
        self.FPS = 3 #3 FRAMES PER SECOND
        self.image = None
        self.quit = False
        self.paused = True
        self.action = 0
        self.treads = [0,0]
        self.speed=.65
        self.timeStart = time.time()
        self.run()


    def getActiveKey(self):
        key = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = event.key
        return key       



    def process_video_from_rover(self, jpegbytes, timestamp_10msec):
        array_of_bytes = np.fromstring(jpegbytes, np.uint8)
        self.image = cv2.imdecode(array_of_bytes, flags=3)
        k = cv2.waitKey(5) & 0xFF
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

                if key in ['w','a','d','s','q', 'z']:
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


            self.clock.tick(self.FPS)
            pygame.display.flip()


        pygame.quit()
        cv2.destroyAllWindows()
        self.close()

