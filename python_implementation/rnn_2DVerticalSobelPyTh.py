__author__ = 'Esha Uboweja'

# This RNN learns a 2-D vertical gradient Sobel (dX) edge detection filter
# [-1, 0, 1
#  -2, 0, 2
#  -1, 0, 1]

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
from RNNBatchPyTh import RNNBatchPyTh

class RNNPyTh2DSobelX(RNNBatchPyTh):

    def genData(self, dataLen):
        """
        Generate data for training / testing the network
        :param dataLen: length of data sequence over time
        :return: x - dataLen x 1 vector of values,
            t - dataLen x 1 vector of containing target results for
            [-1, 0, 1
             -2, 0, 2
             -1, 0, 1]
        """
        # Note: time dimension is along rows, but it corresponds to image ROWS
        M = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).transpose()
        x = np.random.uniform(size=(dataLen, self.nbatches, self.nin))
        t = np.zeros((dataLen, self.nbatches, self.nout))
        for j in xrange(1, dataLen-1):
            for b in xrange(self.nbatches):
                t[j, b, :] = np.multiply(M, x[j-1:j+2, b, :]).sum()
        return x, t

# Number of hidden units
nh = 7
# Number of input units
nin = 3
# Number of output units
nout = 1
# Number of batches
nbatches = 10

# Create RNN
rnnPyTh = RNNPyTh2DSobelX(nh, nin, nout, nbatches)

iters = 50
nepoch = 100
trainLen = 32
step = 0.001
decay = 0.1
errChangeThresh = 10
epochChangeThresh = 10
epochWaitThresh = 10
dataDir = './network_params/'
filePrefix = 'rnn_2DGradientXSobel'
curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

trainLossesNumpy, trainLossesTheano = \
    rnnPyTh.trainNetworkSGD(iters, trainLen, step, decay, errChangeThresh,
                        epochWaitThresh, epochChangeThresh, nepoch, dataDir, filePrefix)

print "Numpy Training error: ", np.mean(np.array(trainLossesNumpy))
print "Theano Training error: ", np.mean(np.array(trainLossesTheano))
fig = plt.figure()
plt.plot(np.arange(1, nepoch + 1), trainLossesNumpy, 'go',
         np.arange(1, nepoch + 1), trainLossesNumpy, 'k')
plt.plot(np.arange(1, nepoch + 1), trainLossesTheano, 'ko',
         np.arange(1, nepoch + 1), trainLossesTheano, 'm')
plt.title('Training losses')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.grid(True)
rnnLossName = dataDir + '/' + filePrefix + '_losses' + '-%d-%d-%d-%s.png' \
                                                       % (
                                                       nh, nin, nout, curTime)
print 'rnnLossName = %s' % rnnLossName
fig.savefig(rnnLossName, dpi=fig.dpi)
plt.show()

# Test the network
testLen = 10000
testErr = rnnPyTh.testNetwork(testLen)
print "Test error: ", testErr

# Image edge detection test
resDir = './network_results/'
M = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).transpose()
s = 32
im = np.zeros((s, s))
s4 = s / 4
im[s4:-s4, s4:-s4] = 1
sx = ndimage.convolve(im, M)
sx = ndimage.sobel(im, axis=1, mode='constant')
res = np.zeros((s, s))
sm = np.zeros((s, s))

th0 = np.zeros((1, rnnPyTh.nh))

for col in xrange(1,s-2):
    colSet = np.reshape(im[:, col-1:col+2], (s, 1, nin))
    _, y = rnnPyTh.forwardPropagation(colSet, th0)
    res[:,col] = np.reshape(y[1:], (s,))
    for row in xrange(1, s-1):
        sm[row-1, col-1] = np.multiply(M, im[row-1:row+2, col-1:col+2]).sum()

fig = plt.figure(figsize=(16,5))
curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filePrefix = 'rnn_2DGradientX_XSq'
rnnName = resDir + filePrefix + '-%d-%d-%d-%s.png' % (nh, nin, nout, curTime)
plt.subplot(141)
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Input square', fontsize=20)
plt.subplot(142)
plt.imshow(sx, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Sobel in X direction', fontsize=20)
plt.subplot(143)
plt.imshow(sm, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Computed in X direction', fontsize=20)
plt.subplot(144)
plt.imshow(res, cmap=plt.cm.gray)
plt.axis('off')
plt.title('RNN test output', fontsize=20)
plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=0.9)
fig.savefig(rnnName, dpi=fig.dpi)
plt.show()
