__author__ = 'Esha Uboweja'

# This RNN learns a 2-D vertical gradient (dX) edge detection filter
# [-1, 0, 1
#  -1, 0, 1
#  -1, 0, 1]

import numpy as np
from RNNTheanoBatch import RNNTheanoBatch
from datetime import datetime
import time
import matplotlib.pyplot as plt
from scipy import ndimage

class RNNTheano2DGradientX(RNNTheanoBatch):

    def genData(self, dataLen):
        """
        Generate data for training / testing the network
        :param dataLen: length of data sequence over time
        :return: x - dataLen x 1 vector of values,
            t - dataLen x 1 vector of containing target results for
            [-1, 0, 1
             -1, 0, 1
             -1, 0, 1]
        """
        # Note: time dimension is along rows, but it corresponds to image ROWS
        M = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).transpose()
        x = np.random.uniform(size=(dataLen, self.nbatches, self.nin))
        t = np.zeros((dataLen, self.nbatches, self.nout))
        for j in xrange(1, dataLen-1):
            for b in xrange(self.nbatches):
                t[j, b, :] = np.multiply(M, x[j-1:j+2, b, :]).sum()
        return x, t


# Number of hidden units
nh = 6
# Number of input units
nin = 3
# Number of output units
nout = 1
# Number of batches
nbatches = 10

# Create RNN using the RNNTheano framework
rnnTheano = RNNTheano2DGradientX(nh, nin, nout, nbatches)

# Train and save the network
iters = 5000
trainLen = 32
step = 0.001
decay = 0.1
errChangeThresh = 10
dataDir = './network_params/'
trainStart = time.time()
trainErr = rnnTheano.trainNetwork(iters, trainLen, step, decay, errChangeThresh,
                                  dataDir, 'rnn_2DGradientX')
trainEnd = time.time()
trainTime = (trainEnd - trainStart) * 1000

avgTrainErr = np.mean(trainErr)
print "Training time: ", trainTime, " (ms), training error: ", avgTrainErr

# Plot training Error
resDir = './network_results'
curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filePrefix = 'rnn_2DGradientX_trainErr'
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
M = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).transpose()
s = 32
im = np.zeros((s, s))
s4 = s / 4
im[s4:-s4, s4:-s4] = 1
sx = ndimage.convolve(im, M)
res = np.zeros((s, s))
sm = np.zeros((s, s))
for col in xrange(1,s-2):
    colSet = np.reshape(im[:, col-1:col+2], (s, 1, nin))
    _, y = rnnTheano.train_fn(np.zeros((1, rnnTheano.nh)), colSet,
                              np.zeros((s, 1, 1)), 0)
    res[:,col] = np.reshape(y, (s,))
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
plt.title('Convolved in X direction', fontsize=20)
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



