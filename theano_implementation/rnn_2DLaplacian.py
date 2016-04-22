__author__ = 'Esha Uboweja'

__author__ = 'Esha Uboweja'

# This RNN learns a 2-D discrete Laplacian edge detection filter
# [ 0,  1, 0
#   1, -4, 1
#   0,  1, 0]

import numpy as np
from RNNTheanoBatch import RNNTheanoBatch
from datetime import datetime
import time
import matplotlib.pyplot as plt
from scipy import ndimage

class RNNTheano2DLaplacian(RNNTheanoBatch):

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

    def imageTest(self, im):
        s = im.shape[0]
        res = np.zeros((s, s))
        sm = np.zeros((s, s))
        for col in xrange(1, s - 2):
            colSet = np.reshape(im[:, col-1:col+2], (s, 1, nin))
            _, y = self.train_fn(np.zeros((1, self.nh)), colSet,
                                      np.zeros((s, 1, 1)), 0)
            res[:, col] = np.reshape(y, (s,))
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
nbatches = 100

# Create RNN using the RNNTheano framework
rnnTheano = RNNTheano2DLaplacian(nh, nin, nout, nbatches)

# Train and save the network
iters = 10000
trainLen = 32
step = 0.00001
decay = 0.1
errChangeThresh = 100
dataDir = './network_params/'
trainStart = time.time()
trainErr = rnnTheano.trainNetwork(iters, trainLen, step, decay, errChangeThresh,
                                  dataDir, 'rnn_2DLaplacian')
trainEnd = time.time()
trainTime = (trainEnd - trainStart) * 1000

avgTrainErr = np.mean(trainErr)
print "Training time: ", trainTime, " (ms), training error: ", avgTrainErr

# Plot training Error
resDir = './network_results/'
curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filePrefix = 'rnn_2DLaplacian_trainErr'
rnnName = resDir + filePrefix + '-%d-%d-%d-%s.png' % (nh, nin, nout, curTime)
epochs = np.arange(1, len(trainErr) + 1)
fig = plt.figure()
plt.plot(epochs, trainErr, 'go', epochs, trainErr, 'k')
plt.ylabel('Training Error per Epoch')
plt.xlabel('Epoch')
plt.title('RNN Training Error over Epochs, time : %f ms' % trainTime)
plt.grid(True)
fig.savefig(rnnName, dpi=fig.dpi)
plt.show()

# Test the network
testLen = 10000
testErr = rnnTheano.testNetwork(testLen)
print "Test error: ", testErr

# Image edge detection test
M = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).transpose()
s = 32
im = np.zeros((s, s))
s4 = s / 4
im[s4:-s4, s4:-s4] = 1
sx = ndimage.convolve(im, M)

res, sm = rnnTheano.imageTest(im)
filePrefix = 'rnn_2DLaplacian'
imageTestPlot(rnnTheano, im, sx, sx, res, resDir, filePrefix)
