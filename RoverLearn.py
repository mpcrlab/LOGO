import numpy as np
import h5py
import torch
import torch.nn.functional as f
from scipy.misc import imresize, bytescale, imrotate
import cv2
import os, sys
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim

D = h5py.File('/home/LOGO/data/rover_dicts.h5', 'r')
D_2 = torch.from_numpy(np.asarray(D['D2'])).cuda(0)
D = torch.from_numpy(np.asarray(D['D'])).cuda(0)

os.chdir('/home/LOGO/data')
files = glob.glob('*Sheri.h5')
print(files)

ps = 15
batch_sz = 250
dst = 2
k, k2 = D.size(1), D_2.size(1)
imsz = np.asarray([240//2, 320//2])
num_rows, num_cols = imsz
#a = torch.zeros(k, num_rows*num_cols)
#a_2 = torch.zeros(k2, num_rows*num_cols)
#phi = torch.randn(1, k).cuda(0)
phi2 = torch.randn(1, k2).cuda(0)



def whiten(X):
    '''Function to ZCA whiten image matrix.'''
    try:
        U,S,V = torch.svd(torch.mm(X, torch.t(X)))
    except RuntimeError:
        return X
    epsilon = 1e-5
    ZCAMatrix = torch.diag(1.0/torch.sqrt(S + epsilon))
    ZCAMatrix = torch.mm(U, torch.mm(ZCAMatrix, torch.t(U)))

    return torch.mm(ZCAMatrix, X)

#
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c1 = nn.Conv2d()

    #
    # # prepare x, normalize, whiten, etc.
    # x = (torch.from_numpy(x).float().cuda(0)).unfold(0,
    #      ps, 1).unfold(1, ps, 1).unfold(2, 3, 1)
    # x = x.contiguous().view(x.size(0)*x.size(1)*x.size(2),
    #                         x.size(3)*x.size(4), x.size(-1))
    # x = x - torch.mean(x, 0)
    # x = whiten(torch.t(x.view(-1, x.size(1)*3)))


h5file = h5py.File(files[0], 'r')
X = np.asarray(h5file['X'])[200, ...]

def get_code(x):
    X = imresize(x, imsz)
    X = np.pad(X, ((ps//2, ps//2),
                   (ps//2, ps//2),
                   (0, 0)), 'reflect')
    X = torch.from_numpy(X).float().cuda(0)
    X = X.unfold(0, ps, 1).unfold(1, ps, 1).unfold(2, 3, 1)
    X = X.contiguous().view(X.size(0)*X.size(1)*X.size(2),
                        X.size(3)*X.size(4), X.size(-1))
    X = X - torch.mean(X, 0)
    X = torch.t(X.view(-1, X.size(1)*3))
    X = whiten(X)
    X = torch.mm(torch.t(D), X)
    X2 = X
    X = X - torch.mean(X, 1)[:, None]
    X = whiten(X)
    X = torch.mm(torch.t(D_2), X)
    #vec = torch.zeros(X.size(1))
    #vec[:num_rows*num_cols] = torch.mm(phi, X2)
    #vec[num_rows*num_cols:] = torch.mm(phi2, X)
    #kk = 0
    fig = plt.figure()
    a1 = fig.add_subplot(121)
    a2 = fig.add_subplot(122)
    a1.imshow(X[0, :].contiguous().view(num_rows, num_cols), cmap='gray')
    a2.imshow(x)
    plt.show()

    # for n in xrange(k):
    #     im = torch.zeros([num_rows//dst, num_cols//dst]).cuda(0)
    #     for i in range(num_rows//dst):
    #         for j in range(num_cols//dst):
    #             im[i, j] = X[n, i*dst*num_cols+j*dst]
    #
    #     vec[n] = torch.sum(phi[..., n] * im)

    # for n in xrange(k2):
    #     im = torch.zeros(num_rows//dst, num_cols//dst).cuda(0)
    #     im2 = torch.zeros(num_rows//dst, num_cols//dst).cuda(0)
    #     for i in range(num_rows//dst):
    #         for j in range(num_cols//dst):
    #             if kk < k:
    #                 im[i, j] = X2[n, i*dst*num_cols+j*dst]
    #             im2[i, j] = X[n, i*dst*num_cols+j*dst]
    #
    #     if kk < k:
    #         vec[n] = torch.sum(phi[..., n] * im)
    #     vec[k+n] = torch.sum(phi[..., k+n] * im2)
    #     kk += 1

        # plt.hist(im.flatten(), bins=100)
        # plt.show()
        #cv2.imshow('rec', np.uint8(bytescale(im-np.mean(im))))
        #cv2.waitKey(500)
        #cv2.imshow('image', imresize(x, imsz))
        #cv2.waitKey(500)
    return torch.max(X, 1)[0]


def get_batch(indx, batch_n):
    h5file = h5py.File(files[indx], 'r')
    X = np.asarray(h5file['X'])
    Y = np.asarray(h5file['Y'])
    rn = np.random.randint(0, X.shape[0], batch_n)
    X, Y = X[rn, ...], torch.from_numpy(np.float32(Y[rn] + 1.)).cuda()
    vecs = torch.zeros(batch_n, k2).cuda()
    #vecs = torch.zeros(batch_sz, k2).cuda()

    for i in range(X.shape[0]):
        vecs[i, :] = get_code(X[i, ...])
    vecs = vecs - torch.mean(vecs, 1)[:, None]

    return Variable(vecs).float(), Variable(Y).long()


def loss_acc(pred, label):
    l = loss_func(pred, label).cuda()
    pr = pred.data.cpu().numpy()
    la = label.data.cpu().numpy()
    a = np.mean(np.int32(np.equal(np.argmax(pr, 1), la)))
    pr = np.argmax(pr, 1)
    conf = np.zeros([4, 4, 3])
    for i in xrange(4):
        for j in xrange(4):
            conf[i, j, :] = np.sum(np.logical_and((la == i), (pr == j)))

    return l, a, conf


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.DataParallel(nn.Linear(k2, 500).cuda())
        self.do1 = nn.Dropout(p=0.4)
        self.fc2 = nn.DataParallel(nn.Linear(500, 500).cuda())
        self.do2 = nn.Dropout(p=0.4)
        self.fc3 = nn.DataParallel(nn.Linear(500, 500).cuda())
        self.do3 = nn.Dropout(p=0.4)
        self.fc6 = nn.DataParallel(nn.Linear(500, 4).cuda())

    def forward(self, x):
        x = self.do1(f.tanh(self.fc1(x)))
        x = self.do2(f.tanh(self.fc2(x)))
        x = self.do3(f.tanh(self.fc3(x)))
        #x = self.do4(f.tanh(self.fc4(x)))
        #x = self.do5(f.tanh(self.fc5(x)))
        x = self.fc6(x)
        return f.softmax(x, dim=1)



writer = SummaryWriter('runs/train')
writer2 = SummaryWriter('runs/test')
writer3 = SummaryWriter('runs/confusion')

Network = NN()
loss_func = nn.CrossEntropyLoss().cuda()
opt = optim.Adam(Network.parameters(), lr=1e-4)

for iters in range(1000):
    X, Y = get_batch(np.random.randint(0, len(files)-2, 1)[0], batch_sz)
    loss, acc, tcm = loss_acc(Network(X), Y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if iters % 20 == 0:
        VX, VY = get_batch(-1, 700)
        VX = Network(VX)
        vl, va, vcm = loss_acc(VX, VY)
        writer.add_scalar('Val. Acc.', va, iters)
        writer2.add_scalar('Val. Loss', vl.data[0], iters)
        writer3.add_image('Val Confusion Matrix', vcm, iters)
        print(vcm)


    writer.add_scalar('Train Acc.', acc, iters)
    writer2.add_scalar('Train Loss', loss.data[0], iters)
    writer3.add_image('Train Confusion Matrix', tcm, iters)
