__author__ = 'Esha Uboweja'

import abc
import numpy as np
import theano
import theano.tensor as TT
from datetime import datetime

class RNNTheanoBatch(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, nh, nin, nout, nbatches):
        """
        RNN setup using Theano tensor data structures
        :param nh: number of hidden units
        :param nin: number of input units
        :param nout: number of output units
        :param nbatches: number of batches
        """
        # Number of hidden units
        self.nh = nh
        # Number of input units
        self.nin = nin
        # Number of output units
        self.nout = nout
        # Number of batches
        self.nbatches = nbatches
        # Input (first dimension is time)
        # Size: [time, batch, features]
        self.x = TT.tensor3()
        # Target (first dimension is time)
        # Size: [time, batch, features]
        self.t = TT.tensor3()
        # Initial hidden state of RNNs
        # Size: [batch, features]
        self.h0 = TT.matrix()
        # Learning rate
        self.lr = TT.scalar()
        # Recurrent hidden node weights
        self.W_hh = theano.shared(np.random.uniform(size=(nh, nh),
                                                    low=-0.01,
                                                    high=0.01))
        # Input layer to hidden layer weights
        self.W_xh = theano.shared(np.random.uniform(size=(nin, nh),
                                                    low=-0.01,
                                                    high=0.01))
        # Hidden layer to output layer weights
        self.W_hy = theano.shared(np.random.uniform(size=(nh, nout),
                                                    low=-0.01,
                                                    high=0.01))

        # Compute hidden state and output for the entire input sequence
        # (first dimension is time)
        [self.h, self.y], _ = theano.scan(self.step,
                                          sequences = self.x,
                                          outputs_info=[self.h0, None],
                                          non_sequences=[self.W_hh, self.W_xh,
                                                         self.W_hy])

        # Error between output and target
        self.error = 0.5 * ((self.y - self.t) ** 2).sum().sum()

        # BPTT (back-propagation through time)
        # Gradients
        self.gW_hh, self.gW_xh, self.gW_hy = TT.grad(self.error,
                                                     [self.W_hh, self.W_xh,
                                                      self.W_hy])
        print "Gradient shapes: gW_hh: ", self.gW_hh.shape, ", gW_xh: ", \
            self.gW_xh.shape, ", gW_hy: ", self.gW_hy.shape
        # Training function
        self.train_fn = theano.function(
            [self.h0, self.x, self.t, self.lr],
            [self.error, self.y],
            updates={self.W_hh: self.W_hh - self.lr * self.gW_hh,
                     self.W_xh: self.W_xh - self.lr * self.gW_xh,
                     self.W_hy: self.W_hy - self.lr * self.gW_hy})

    def step(self, x_t, h_tm1, W_hh, W_xh, W_hy):
        """
        Forward step function (recurrent)
        Nonlinear activation function - one of tanh, sigmoid, ReLU
        :param x_t: input at time-step t
        :param h_tm1: hidden state at time-step (t-1)
        :return: h_t: updated hidden state at time-step t
                 y_t: updated output at time-step t
        """
        h_t = TT.tanh(TT.dot(x_t, W_xh) + TT.dot(h_tm1, W_hh))
        y_t = TT.tanh(TT.dot(h_t, W_hy))
        return h_t, y_t

    @abc.abstractmethod
    def genData(self, dataLen):
        """
        Generate data for training / testing the network
        :param dataLen: length of data sequence over time
        :param nin: number of input units
        :return: x - dataLen x 1 vector of values,
                 t - dataLen x 1 vector of containing target results
        """
        return

    def saveNetwork(self, paramFile):
        """
        Save network parameters, weights
        :return: (Nothing)
        """
        W_hh, W_xh, W_hy = self.W_hh.get_value(), self.W_xh.get_value(), \
                           self.W_hy.get_value()
        np.savez(paramFile, W_hh=W_hh, W_xh=W_xh, W_hy=W_hy)

    def trainNetwork(self, iters, dataLen, step, decay, errChangeThresh,
                     dataDir, filePrefix):
        """
        Train the RNN
        :param rnn: RNN network object to train on
        :param iters: number of iterations for training the network
        :param dataLen: number of timesteps in each input vector
        :param step: learning rate value
        :param decay: decay rate for learning rate
        :param errChangeThresh: threshold to decide decay of learning rate
        :param dataDir: directory to save the weight matrices
        :param filePrefix: file prefix name for saving parameters
        :return: - weight vectors are stored in the class object for RNN
                 error: mean training error
        """

        error = []
        changedIter = 500
        changeIter = 0
        iterThresh = 1000
        for i in xrange(iters):
            # Generate training data
            x, t = self.genData(dataLen)
            # Train the network
            err, y = self.train_fn(np.zeros((self.nbatches, self.nh)), x, t, step)
            error += [err]
            print "Error for iter: ", i, " is ", err, ", meanErr: ", \
                    err / (y.shape[0] * y.shape[1])
            changeIter += 1
            # Decay learning rate if loss didn't change in last 100 iterations
            if changeIter >= changedIter and i >= iterThresh:
                changeIter = 0
                last100Err = np.mean(np.array(error[i-iterThresh:i]))
                print "abs(err - last100Err): ", abs(err - last100Err)
                if (abs(err - last100Err) <= errChangeThresh):
                    step *= decay
                    print "New LEARNING rate: ", step
        curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        rnnName = dataDir + filePrefix + '-%d-%d-%d-%s.npz' % \
                            (self.nh, self.nin, self.nout, curTime)
        self.saveNetwork(rnnName)
        return np.array(error)

    def testNetwork(self, dataLen):
        """
        Test the network with a new input sequence
        :param rnn: RNN object, with weight parameters
        :param dataLen: number of timesteps in each input vector
        :return: err - testing error
        """
        # Generate test data
        x, t = self.genData(dataLen)
        # Test the network
        _, y = self.train_fn(np.zeros((self.nbatches, self.nh)), x,
                             np.zeros((dataLen, self.nbatches, self.nout)), 0)
        # Compute the error between computed output and actual target
        err = 0.5 * ((y - t) ** 2).sum().sum().sum() / (dataLen * self.nbatches)
        return err
