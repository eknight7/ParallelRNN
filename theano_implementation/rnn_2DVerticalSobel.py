__author__ = 'Esha Uboweja'

# This RNN learns a 2-D vertical Sobel edge detection filter
# [-1, 0, 1
#  -2, 0, 1
#  -1, 0, 1]

import numpy as np
from RNNTheano import RNNTheano
import matplotlib.pyplot as plt
from scipy import ndimage

class RNNTheano2DSobelX(RNNTheano):

    def genData(self, dataLen):
        """
        Generate data for training / testing the network
        :param dataLen: length of data sequence over time
        :return: x - dataLen x 1 vector of values,
            t - dataLen x 1 vector of containing target results for [-1, 0, 1]
        """
        # Note: time dimension is along rows, but it corresponds to image columns
        m = np.array([1, 2, 1])
        x = np.random.uniform(size=(dataLen, self.nin))
        t = [np.dot(m, x[j+2,:]) - np.dot(m, x[j,:]) for j in range(dataLen-2)]
        t = np.reshape(np.asarray([0] + t + [0]), (dataLen, 1))
        return x, t


# Number of hidden units
nh = 30
# Number of input units
nin = 3
# Number of output units
nout = 1

# Create RNN using the RNNTheano framework
rnnTheano = RNNTheano2DSobelX(nh, nin, nout)

# Train and save the network
iters = 1000000
trainLen = 32
step = 0.0001
dataDir = './network_params/'
trainErr = rnnTheano.trainNetwork(iters, trainLen, step,
                                  dataDir, 'rnn_2DVerticalSobel')
print "Training error: ", trainErr

# Test the network
testLen = 10000
testErr = rnnTheano.testNetwork(testLen)
print "Test error: ", testErr

# Image edge detection test
s = 32
im = np.zeros((s, s))
s4 = s / 4
im[s4:-s4, s4:-s4] = 1
sx = ndimage.sobel(im, axis=0, mode='constant')
res = np.zeros((s, s))
for row in xrange(1, s-1):
    rowSet = np.reshape(im[row-1:row+2, :], (s, 3))
    _, y = rnnTheano.train_fn(np.zeros((rnnTheano.nh,)), rowSet,
                              np.zeros((s, 1)), 0)
    res[row,:] = np.reshape(y, (1, s))
plt.figure(figsize=(16,5))
plt.subplot(131)
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Input square', fontsize=20)
plt.subplot(132)
plt.imshow(sx, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Sobel output, sx', fontsize=20)
plt.subplot(133)
plt.imshow(res, cmap=plt.cm.gray)
plt.axis('off')
plt.title('RNN test output', fontsize=20)
plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=0.9)
plt.show()


