__author__ = 'Esha Uboweja'

import abc
import numpy as np
from datetime import datetime

class RNNBatchPy(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, nh, nin, nout, nbatches):
        """
        RNN setup using Numpy arrays
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
        # Learning rate, default to 0.01
        self.lr = 0.01
        # Recurrent hidden node weights
        self.W_hh = np.random.uniform(size=(nh, nh), low=-0.01, high=0.01)
        # Input layer to hidden layer weights
        self.W_xh = np.random.uniform(size=(nin, nh), low=-0.01, high=0.01)
        # Hidden layer to output layer weights
        self.W_hy = np.random.uniform(size=(nh, nout), low=-0.01, high=0.01)

    def forwardPropagation(self, x, h0):
        """
        Forward propagation function (recurrent)
        :param x: input of size T x nbatches x nin, first dimension is time
        :param h0: initial hidden state, of nh units, size nbatches x nh
        :return: h: resulting hidden state for all nodes using input x
                    for every time-step, h : T x nbatches x nh
                 y: resulting output using current weights for input x
                    for every time-step, y : T x nbatches x nout
        """
        # Check hidden state dim
        if h0.shape[1] != self.nh:
            raise ValueError("Invalid shape of initial hidden state vector h0")
        # Check input dim
        if (self.nin > 1 and len(x.shape) != 3) or \
                (len(x.shape) == 3 and x.shape[2] != self.nin):
            raise ValueError("Input dimension should be Time x Batches x #Input Dim")
        # Number of time-steps
        T = x.shape[0]

        nbatches = h0.shape[0]
        # Starting at step t = 1, up to step T, update the hidden state
        # and the output at each step
        # Hidden state at every time-step
        h = np.zeros((T + 1, nbatches, self.nh))
        h[0] = h0
        # Output: T x nbatches x nout, first dimension is time
        y = np.zeros((T + 1, nbatches, self.nout))
        for t in xrange(1, T + 1):
            # Update hidden state at t
            h[t] = np.tanh(np.dot(x[t-1,], self.W_xh) + np.dot(h[t-1], self.W_hh))
            # Update output at t
            y[t] = np.tanh(np.dot(h[t], self.W_hy))

        return h, y

    def calculateLoss(self, x, t, h0):
        """
        Computes the error or the loss when we train the RNN with the input
        x, with target output t, and initial hidden state h0
        :param x: input of size T x nbatches x nin, first dimension is time
        :param t: target output of size T x nbatches x nout, first dimension is time
        :param h0: initial hidden state, of nh units
        :return: error: the loss value after training RNN with input x, target t
        """
        # Check input dim
        if (self.nin > 1 and len(x.shape) != 3) or \
                (len(x.shape) == 3 and x.shape[2] != self.nin):
            raise ValueError(
                "Input dimension should be Time x Batches x #Input Dim")
        # Check target dim
        if (self.nout > 1 and len(t.shape) != 3) or \
                (len(t.shape) == 2 and t.shape[2] != self.nout):
            raise ValueError(
                "Target dimension should be Time x Batches x #Output Dim")
        # Check hidden state dim
        if h0.shape[1] != self.nh:
            raise ValueError("Invalid shape of initial hidden state vector h0")

        # Do the forward propagation and compute output
        _, y = self.forwardPropagation(x, h0)
        # Compute error between target and computed output
        error = 0.5 * ((y[1:, :] - t) ** 2).sum().sum()
        return error

    def bptt(self, x, t, h0):
        """
        Computes the total gradient using back-propagation through time (BPTT)
        when we train the RNN with input x, with target output t, and initial
        hidden state h0
        :param x: input of size T x batches x nin, first dimension is time
        :param t: target output of size T x batches x nout, first dimension is time
        :param h0: initial hidden state, of nh units, size batches x nh
        :return: dE_xh: total gradient of W_xh w.r.t. updates using x
                 dE_hh: total gradient of W_hh w.r.t. updates using x
                 dE_hy: total gradient of W_hy w.r.t. updates using x
        """
        # Check input dim
        if (self.nin > 1 and len(x.shape) != 3) or \
                (len(x.shape) == 3 and x.shape[2] != self.nin):
            raise ValueError(
                "Input dimension should be Time x Batches x #Input Dim")
        # Check target dim
        if (self.nout > 1 and len(t.shape) != 3) or \
                (len(t.shape) == 2 and t.shape[2] != self.nout):
            raise ValueError(
                "Target dimension should be Time x Batches x #Output Dim")
        # Check hidden state dim
        if h0.shape[1] != self.nh:
            raise ValueError(
                "Invalid shape of initial hidden state vector h0")

        # Do the forward propagation and compute output
        h, y = self.forwardPropagation(x, h0)

        # Compute total gradients
        dE_xh = np.zeros((self.nin, self.nh))
        dE_hh = np.zeros((self.nh, self.nh))
        # Number of time-steps
        T = x.shape[0]
        # (y - t) ^ 2 per element
        targetDiff = (y[1:, :] - t)
        # (1 - y^2) per element
        outputDeriv = (1 - y[1:, :] ** 2)
        # Delta on output
        deltaOutput = targetDiff * outputDeriv
        # Derivative of loss w.r.t. W_hy
        # dE_hy : nh x nout
        dE_hy = np.tensordot(h[1:, :], deltaOutput, axes=([0, 1], [0, 1]))

        hstep = (1 - h ** 2)
        dhh = np.zeros((T + 1, self.nbatches, self.nh))

        for tstep in xrange(T, 0, -1):
            # dE_t/dy_t * dy_t/dh_t
            dhh[tstep] += np.dot(deltaOutput[tstep - 1], self.W_hy.T)
            r = (dhh[tstep] * hstep[tstep])
            dhh[tstep - 1] += np.dot(r, self.W_hh.T)

            dE_xh += np.dot(x[tstep - 1, :].T, r)
            dE_hh += np.dot(h[tstep - 1, :].T, r)

        return (dE_xh, dE_hh, dE_hy)

    def sgdStep(self, x, t, h0):
        """
        Update the weights for each connection using BPTT in one SGD step
        :param x: input of size T x nbatches x nin, first dimension is time
        :param t: target output of size T x nbatches x nout, first dimension is time
        :param h0: initial hidden state, of nh units, h0: nbatches x nout
        :return: (nothing)
        """
        # Calculate gradients using BPTT
        dE_xh, dE_hh, dE_hy = self.bptt(x, t, h0)

        # Update weights based on gradients
        self.W_xh -= self.lr * dE_xh
        self.W_hh -= self.lr * dE_hh
        self.W_hy -= self.lr * dE_hy

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
        np.savez(paramFile, W_hh=self.W_hh, W_xh=self.W_xh, W_hy=self.W_hy)

    def trainNetworkSGD(self, iters, dataLen, step, decay, errChangeThresh,
                        epochWaitThresh, epochChangeThresh, nepoch,
                        dataDir, filePrefix):
        """
        Train the RNN using SGD
        :param iters: number of iterations for training the network
        :param dataLen: number of time-steps in each input vector
        :param step: learning rate value
        :param decay: decay rate for learning rate
        :param errChangeThresh: threshold to decide decay of learning rate
        :param epochChangeThresh: number of epochs to observe loss for
                                  changing learning rate if needed
        :param epochWaitThresh: number of epochs to wait before
                                  changing learning rate if needed
        :param nepoch: Number of epoch evaluations of training dataset
        :param dataDir: directory to save the weight matrices
        :param filePrefix: file prefix name for saving parameters
        :return: - weight vectors are stored in the class object for RNN
                   losses: mean training loss after each epoch
        """
        lossesNumpy = []

        self.lr = step
        h0 = np.zeros((self.nbatches, self.nh))
        # For decay rate purposes
        epochWaitCount = 0
        for epoch in xrange(nepoch):

            # Total loss in current epoch
            totalLossNumpy = 0
            for idx in xrange(iters):
                # Generate training data
                x, t = self.genData(dataLen)
                # Calculate loss using this bit of data
                loss = self.calculateLoss(x, t, h0)
                totalLossNumpy += loss
                # Train the network, take a step in the direction of maximum
                # gradient, using SGD
                self.sgdStep(x, t, h0)

            totalLossNumpy /= iters

            lossesNumpy += [totalLossNumpy]
            print "Epoch = %d, numpy loss = %f, mean error = %f" \
                  % (epoch, totalLossNumpy, totalLossNumpy / (self.nbatches * self.nh))

            # Decay learning rate if loss hasn't changed in past
            if epochWaitCount >= epochWaitThresh and epoch >= epochChangeThresh:
                # Reset
                epochWaitCount = 0
                # Check loss for last few epochs
                prevErr = np.mean(np.array(
                            lossesNumpy[epoch-epochChangeThresh:epoch]))
                err = abs(lossesNumpy[epoch] - prevErr)
                print "err: ", err
                if (err <= errChangeThresh):
                    step *= decay
                    print "Epoch = %d, new learning rate: %f" % (epoch, step)

            epochWaitCount += 1

        curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        rnnName = dataDir + '/' + filePrefix + '-%d-%d-%d-%s.npz' % \
                                               (self.nh, self.nin, self.nout,
                                                curTime)
        self.saveNetwork(rnnName)
        return lossesNumpy

    def testNetwork(self, dataLen):
        """
        Test the network with a new input sequence
        :param rnn: RNN object, with weight parameters
        :param dataLen: number of timesteps in each input vector
        :return: err - testing error
        """
        # Generate test data
        x, t = self.genData(dataLen)
        h0 = np.zeros((self.nbatches, self.nh))
        # Test the network
        _, y = self.forwardPropagation(x, h0)
        # Compute the error between computed output and actual target
        err = ((y[1:, :] - t) ** 2).sum().sum() / dataLen
        return err
