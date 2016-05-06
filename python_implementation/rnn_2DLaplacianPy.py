__author__ = 'Esha Uboweja'

# This RNN learns a 2-D discrete Laplacian edge detection filter
# [ 0,  1, 0
#   1, -4, 1
#   0,  1, 0]

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
from RNNBatchPy import RNNBatchPy
import time

class RNNPy2DLaplacian(RNNBatchPy):

    def genData(self, dataLen):
        """
        Generate data for training / testing the network
        :param dataLen: length of data sequence over time
        :return: x - dataLen x 1 vector of values,
            t - dataLen x 1 vector of containing target results for
            [ 0,  1, 0
              1, -4, 1
              0,  1, 0]
        """
        # Note: time dimension is along rows, but it corresponds to image ROWS
        M = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).transpose()
        x = np.random.uniform(size=(dataLen, self.nbatches, self.nin))
        t = np.zeros((dataLen, self.nbatches, self.nout))
        for j in xrange(1, dataLen-1):
            for b in xrange(self.nbatches):
                t[j, b, :] = np.multiply(M, x[j-1:j+2, b, :]).sum()
        return x, t

    def imageTest(self, im, M):
        s = im.shape[0]
        res = np.zeros((s, s))
        sm = np.zeros((s, s))

        th0 = np.zeros((1, self.nh))
        for col in xrange(1, s - 2):
            colSet = np.reshape(im[:, col - 1:col + 2], (s, 1, self.nin))
            _, y = self.forwardPropagation(colSet, th0)
            res[:, col] = np.reshape(y[1:], (s,))
            for row in xrange(1, s - 1):
                sm[row - 1, col - 1] = np.multiply(M, im[row - 1:row + 2,
                                                      col - 1:col + 2]).sum()
        return res, sm

def imageTestPlot(rnn, im, convRes, compRes, result, resDir, filePrefix):
    fig = plt.figure(figsize=(16, 5))
    curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    rnnName = resDir + filePrefix + '-%d-%d-%d-%s.png' \
                                    % (rnn.nh, rnn.nin, rnn.nout, curTime)
    plt.subplot(141)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Input', fontsize=20)
    plt.subplot(142)
    plt.imshow(convRes, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Convolved Laplacian', fontsize=20)
    plt.subplot(143)
    plt.imshow(compRes, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Computed Laplacian', fontsize=20)
    plt.subplot(144)
    plt.imshow(result, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('RNN test output', fontsize=20)
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0,
                        right=0.9)
    fig.savefig(rnnName, dpi=fig.dpi)
    plt.show()

# Number of hidden units
nh = 12
# Number of input units
nin = 3
# Number of output units
nout = 1
# Number of batches
nbatches = 200

# Create RNN
rnnPyTh = RNNPy2DLaplacian(nh, nin, nout, nbatches)

iters = 100
nepoch = 100
trainLen = 32
step = 0.0001
decay = 0.1
errChangeThresh = 100
epochChangeThresh = 10
epochWaitThresh = 5
dataDir = './network_params/'
filePrefix = 'rnn_2DGradientLaplacian'
curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

startTime = time.time()
trainLossesNumpy = \
    rnnPyTh.trainNetworkSGD(iters, trainLen, step, decay, errChangeThresh,
                epochWaitThresh, epochChangeThresh, nepoch, dataDir, filePrefix)
endTime = time.time()

print "Numpy Training error: ", np.mean(np.array(trainLossesNumpy))
print "Numpy Training time: ", (endTime - startTime) * 1000, " ms"

fig = plt.figure()
plt.plot(np.arange(1, nepoch + 1), trainLossesNumpy, 'go',
         np.arange(1, nepoch + 1), trainLossesNumpy, 'k')
plt.title('Training losses')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.grid(True)
rnnLossName = dataDir + '/' + filePrefix + '_losses' + '-%d-%d-%d-%s.png' \
                                               % (nh, nin, nout, curTime)
print 'rnnLossName = %s' % rnnLossName
fig.savefig(rnnLossName, dpi=fig.dpi)
plt.show()

# Test the network
testLen = 10000
testErr = rnnPyTh.testNetwork(testLen)
print "Test error: ", testErr

# Image edge detection test
resDir = './network_results/'
M = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).transpose()
s = 32
im = np.zeros((s, s))
s4 = s / 4
im[s4:-s4, s4:-s4] = 1
sx = ndimage.convolve(im, M)

res, sm = rnnPyTh.imageTest(im, M)
filePrefix = 'rnn_2DLaplacianSq'
imageTestPlot(rnnPyTh, im, sx, sm, res, resDir, filePrefix)

dataDir = '../data/'
imList = ['airliner_s_001152.png', 'automobile_s_000220.png',
          'monoplane_s_001328.png', 'riding_horse_s_000148.png']
for imname in imList:
    im = ndimage.imread(dataDir + imname, 'L')

    sx = ndimage.sobel(im, axis=1, mode='constant')
    res, sm = rnnPyTh.imageTest(im, M)
    filePrefix = 'rnn_2DLaplacian_' + imname.split(".")[0]
    imageTestPlot(rnnPyTh, im, sx, sm, res, resDir, filePrefix)

