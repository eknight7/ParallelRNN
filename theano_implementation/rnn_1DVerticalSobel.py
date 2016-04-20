__author__ = 'Esha Uboweja'

# This RNN learns a 1-D vertical Sobel edge detection filter [-1 0 1]

import numpy as np
import random
from RNNTheano import RNNTheano
from datetime import datetime

# Generate data
def genData(dataLen):
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

# Network training
def trainNetwork(rnn, iters, dataLen, step, dataDir):
    """
    Train the RNN
    :param rnn: RNN network object to train on
    :param iters: number of iterations for training the network
    :param dataLen: number of timesteps in each input vector
    :param step: learning rate value
    :param dataDir: directory to save the weight matrices
    :return: - weight vectors are stored in the class object for RNN
             error: mean training error
    """
    error = 0
    for i in xrange(iters):
        # Generate training data
        x, t = genData(dataLen)
        # Train the network
        err, y = rnn.train_fn(np.zeros(rnn.nh, ), x, t, step)
        error += err
    error /= iters
    saveNetwork(rnnTheano, dataDir)
    return error

# Save the network weights learnt after training
def saveNetwork(rnn, dataDir):
    """
    Save the network parameters
    :param rnn: RNN object, with weight parameters
    :param dataDir: directory to save the weight matrices
    :return: (Nothing)
    """
    curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    rnnName = dataDir + \
              'rnn_1DVerticalSobel-%d-%d-%d-%s.npz' % (nh, nin, nout, curTime)
    rnn.saveNetwork(rnnName)

# Network testing
def testNetwork(rnn, dataLen):
    """
    Test the network with a new input sequence
    :param rnn: RNN object, with weight parameters
    :param dataLen: number of timesteps in each input vector
    :return: err - testing error
    """
    # Generate test data
    x, t = genData(dataLen)
    # Test the network
    _, y = rnn.train_fn(np.zeros(rnn.nh, ), x, np.zeros((dataLen, 1)), 0)
    # Compute the error between computed output and actual target
    err = ((y - t) ** 2).sum() / dataLen
    return err

# Number of hidden units
nh = 4
# Number of input units
nin = 1
# Number of output units
nout = 1

# Create RNN using the RNNTheano framework
rnnTheano = RNNTheano(nh, nin, nout)

# Train and save the network
iters = 1000000
trainLen = 10
step = 0.001
dataDir = './network_params/'
trainErr = trainNetwork(rnnTheano, iters, trainLen, step, dataDir)
print "Training error: ", trainErr

# Test the network
testLen = 10000
testErr = testNetwork(rnnTheano, testLen)
print "Test error: ", testErr
