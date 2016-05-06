__author__ = 'Esha Uboweja'

# This RNN learns a 1-D vertical Prewitt edge detection filter [-1 0 1]

import numpy as np
import random
from RNNPy import RNNPy
from datetime import datetime
import matplotlib.pyplot as plt
import time

class RNNPy1DPrewittX(RNNPy):

    def genData(self, dataLen):
        """
        Generate data for training / testing the network
        :param dataLen: length of data sequence over time
        :return: x - dataLen x 1 vector of values,
                 t - dataLen x 1 vector of containing target results for [-1, 0, 1]
        """
        x = np.reshape(np.asarray([random.random() for _ in xrange(dataLen)]),
                       (dataLen, 1))
        t = [x[j + 2] - x[j] for j in xrange(dataLen - 2)]
        t = np.reshape(np.asarray([0, 0] + t), (dataLen, 1))
        return x, t

# Number of hidden units
nh = 4
# Number of input units
nin = 1
# Number of output units
nout = 1

# Create RNN using RNNPy framework
rnnPy = RNNPy1DPrewittX(nh, nin, nout)

# Train and save the network
iters = 200
nepoch = 1000
trainLen = 10
step = 0.001
dataDir = './network_params'
filePrefix = 'rnn_1DVerticalPrewitt'
curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Perform gradient checking
result = rnnPy.gradientCheck(trainLen)

startTime = time.time()
trainLosses = rnnPy.trainNetworkSGD(iters, trainLen, step, nepoch,
                                    dataDir, filePrefix)
endTime = time.time()
print "Training error: ", np.mean(np.array(trainLosses))
print "Training time: ", (endTime - startTime), " ms"
fig = plt.figure()
plt.plot(np.arange(1, nepoch+1), trainLosses, 'go',
         np.arange(1, nepoch+1), trainLosses, 'k')
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
testErr = rnnPy.testNetwork(testLen)
print "Test error: ", testErr
