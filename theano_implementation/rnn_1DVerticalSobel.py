__author__ = 'Esha Uboweja'

# This RNN learns a 1-D vertical Sobel edge detection filter [-1 0 1]

import numpy as np
import random
from RNNTheano import RNNTheano
import matplotlib.pyplot as plt
import time
from datetime import datetime

class RNNTheano1DSobelX(RNNTheano):

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

# Create RNN using the RNNTheano framework
rnnTheano = RNNTheano1DSobelX(nh, nin, nout)

# Train and save the network
iters = 100
nepoch = 1000
trainLen = 10
step = 0.001
dataDir = './network_params/'
filePrefix= 'rnn_1DVerticalSobel'
startTime = time.time()
trainErr, trainLosses = rnnTheano.trainNetwork(iters, nepoch, trainLen, step,
                                  dataDir, filePrefix)
endTime = time.time()
print "Training error: ", trainErr
print "Training time: ", (endTime - startTime), " ms"

fig = plt.figure()
plt.plot(np.arange(1, nepoch+1), trainLosses, 'go',
         np.arange(1, nepoch+1), trainLosses, 'k')
plt.title('Training losses')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.grid(True)
curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
rnnLossName = dataDir + '/' + filePrefix + '_losses' + '-%d-%d-%d-%s.png' \
                                               % (nh, nin, nout, curTime)
print 'rnnLossName = %s' % rnnLossName
fig.savefig(rnnLossName, dpi=fig.dpi)

# Test the network
testLen = 10000
testErr = rnnTheano.testNetwork(testLen)
print "Test error: ", testErr
