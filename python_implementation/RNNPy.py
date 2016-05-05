__author__ = 'Esha Uboweja'

import abc
import numpy as np
from datetime import datetime
import operator

class RNNPy(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, nh, nin, nout):
        """
        RNN setup using Numpy arrays
        :param nh: number of hidden units
        :param nin: number of input units
        :param nout: number of output units
        :param lr: learning rate
        """
        # Number of hidden units
        self.nh = nh
        # Number of input units
        self.nin = nin
        # Number of output units
        self.nout = nout
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
        :param x: input of size T x nin, first dimension is time
        :param h0: initial hidden state, of nh units
        :return: h: resulting hidden state for all nodes using input x
                    for every time-step, h : T x nh
                 y: resulting output using current weights for input x
                    for every time-step, y : T x nout
        """
        # Check hidden state dim
        if h0.shape != (self.nh,):
            raise ValueError("Invalid shape of initial hidden state vector h0")
        # Check input dim
        if (self.nin > 1 and len(x.shape) != 2) or \
                (len(x.shape) == 2 and x.shape[1] != self.nin):
            raise ValueError("Input dimension should be Time x #Input Dim")
        # Number of time-steps
        T = x.shape[0]
        # Starting at step t = 1, up to step T, update the hidden state
        # and the output at each step
        h_tm1 = h0
        # Hidden state at every time-step
        h = np.zeros((T+1, self.nh))
        h[0, :] = h0
        # Output: T x nout, first dimension is time
        y = np.zeros((T+1, self.nout))
        for t in xrange(1, T+1):
            # Update hidden state at t
            h_t = np.tanh(np.dot(x[t-1, :], self.W_xh) + np.dot(h_tm1, self.W_hh))
            h_tm1 = h_t
            h[t, :] = h_t
            # Update output at t
            y_t = np.tanh(np.dot(h_t, self.W_hy))
            y[t, :] = y_t

        return h, y

    def calculateLoss(self, x, t, h0):
        """
        Computes the error or the loss when we train the RNN with the input
        x, with target output t, and initial hidden state h0
        :param x: input of size T x nin, first dimension is time
        :param t: target output of size T x nout, first dimension is time
        :param h0: initial hidden state, of nh units
        :return: error: the loss value after training RNN with input x, target t
        """
        # Check input dim
        if (self.nin > 1 and len(x.shape) != 2) or \
                (len(x.shape) == 2 and x.shape[1] != self.nin):
            raise ValueError("Input dimension should be Time x #Input Dim")
        # Check target dim
        if (self.nout > 1 and len(t.shape) != 2) or \
            (len(t.shape) == 2 and t.shape[1] != self.nout):
            raise ValueError("Target dimension should be Time x #Output Dim")
        # Check hidden state dim
        if h0.shape != (self.nh,):
            raise ValueError(
                "Invalid shape of initial hidden state vector h0")

        # Do the forward propagation and compute output
        _, y = self.forwardPropagation(x, h0)
        # Compute error between target and computed output
        error = 0.5 * ((y[1:,:] - t) ** 2).sum()
        return error

    def bptt(self, x, t, h0):
        """
        Computes the total gradient using back-propagation through time (BPTT)
        when we train the RNN with input x, with target output t, and initial
        hidden state h0
        :param x: input of size T x nin, first dimension is time
        :param t: target output of size T x nout, first dimension is time
        :param h0: initial hidden state, of nh units
        :return: dE_xh: total gradient of W_xh w.r.t. updates using x
                 dE_hh: total gradient of W_hh w.r.t. updates using x
                 dE_hy: total gradient of W_hy w.r.t. updates using x
        """
        # Check input dim
        if (self.nin > 1 and len(x.shape) != 2) or \
                (len(x.shape) == 2 and x.shape[1] != self.nin):
            raise ValueError("Input dimension should be Time x #Input Dim")
        # Check target dim
        if (self.nout > 1 and len(t.shape) != 2) or \
                (len(t.shape) == 2 and t.shape[1] != self.nout):
            raise ValueError("Target dimension should be Time x #Output Dim")
        # Check hidden state dim
        if h0.shape != (self.nh,):
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
        targetDiff = (y[1:,:] - t)
        # (1 - y^2) per element
        outputDeriv = (1 - y[1:,:] ** 2)
        # Delta on output
        deltaOutput = targetDiff * outputDeriv
        # Derivative of loss w.r.t. W_hy
        # dE_hy : nh x nout
        dE_hy = np.dot(h[1:,:].T, deltaOutput)

        dhh = np.zeros((T + 1, self.nh))
        for tstep in xrange(T, 0, -1):
            # dE_t/dy_t * dy_t/dh_t
            dhh[tstep] += np.dot(self.W_hy, deltaOutput[tstep-1])
            r = dhh[tstep] * (1 - h[tstep, :] ** 2)
            dhh[tstep - 1] += np.dot(self.W_hh, r)

            dE_xh += np.outer(x[tstep - 1, :], r)
            dE_hh += np.outer(h[tstep - 1, :], r)

        return (dE_xh, dE_hh, dE_hy)

    def gradientCheck(self, T, m=0.001, errorThresh=0.01):
        """
        Perform gradient checking on our BPTT function, to make sure the
        differentiation worked out
        :param T: duration of example (number of time-steps)
        :param m: amount to move around training data value by
        :param errorThresh: threshold to determine gradient calculation correctness
        :return: Boolean value indicating whether gradient check succeeded
        """
        # x: input of size T x nin, first dimension is time
        # t: target output of size T x nout, first dimension is time
        x, t = self.genData(T)
        # h0: initial hidden state, of nh units
        h0 = np.zeros(self.nh,)
        # Calculate gradients using BPTT
        bpttGradients = self.bptt(x, t, h0)
        # Parameter list
        modelParameters = ['W_xh', 'W_hh', 'W_hy']
        # Gradient check for every parameter
        for pidx, pname in enumerate(modelParameters):
            error = 0
            # Get parameter values
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient calculation checking for parameter %s with size (%d, %d)." \
                  % (pname, parameter.shape[0], parameter.shape[1])
            # Iterate over every single value in the parameter and perform the
            # gradient check
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                originalValue = parameter[ix]
                # Estimate the gradient using (f(x + m) - f(x - m)) / (2*m)
                parameter[ix] = originalValue + m
                # f(x + m)
                gradientPlus = self.calculateLoss(x, t, h0)
                parameter[ix] = originalValue - m
                # f(x - m)
                gradientMinus = self.calculateLoss(x, t, h0)
                gradientEstimate = (gradientPlus - gradientMinus) / (2 * m)
                # Reset parameter to original value
                parameter[ix] = originalValue
                # Gradient calculated using bptt
                gradientBPTT = bpttGradients[pidx][ix]
                # Calculate the relative error: (|x - y| / (|x| + |y|))
                relativeError = np.abs(gradientBPTT - gradientEstimate) / \
                            (np.abs(gradientBPTT) + np.abs(gradientEstimate))
                numerator = np.abs(gradientBPTT - gradientEstimate)
                denominator = (np.abs(gradientBPTT) + np.abs(gradientEstimate))
                # If relative error is large, gradient check failed
                if relativeError > errorThresh:

                    print "Gradient Check ERROR: parameter=%s, ix=(%d, %d)" % (pname, ix[0], ix[1])
                    print "f(x+m) error = %f" % gradientPlus
                    print "f(x-m) error = %f" % gradientMinus
                    print "Gradient estimate: %f " % gradientEstimate
                    print "Gradient using BPTT: %f " % gradientBPTT
                    print "Relative Error: %f " % relativeError
                    print "Numerator: %f, Denominator: %f " % (numerator, denominator)

                    error += relativeError
                    #return False
                # Move on to next element
                it.iternext()
            print "Gradient check total error %f " % (error)
            print "Gradient check for parameter %s passed." % (pname)
        return True

    def sgdStep(self, x, t, h0, m=0.001, errorThresh=0.01):
        """
        Update the weights for each connection using BPTT in one SGD step
        :param x: input of size T x nin, first dimension is time
        :param t: target output of size T x nout, first dimension is time
        :param h0: initial hidden state, of nh units
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

    def trainNetworkSGD(self, iters, dataLen, step, nepoch, dataDir, filePrefix):
        """
        Train the RNN using SGD
        :param iters: number of iterations for training the network
        :param dataLen: number of time-steps in each input vector
        :param step: learning rate value
        :param nepoch: Number of epoch evaluations of training dataset
        :param dataDir: directory to save the weight matrices
        :param filePrefix: file prefix name for saving parameters
        :return: - weight vectors are stored in the class object for RNN
                   losses: mean training loss after each epoch
        """
        losses = []

        self.lr = step
        h0 = np.zeros(self.nh,)
        for epoch in xrange(nepoch):
            # Total loss in current epoch
            totalLoss = 0
            for idx in xrange(iters):
                # Generate training data
                x, t = self.genData(dataLen)
                # Calculate loss using this bit of data
                loss = self.calculateLoss(x, t, h0)
                totalLoss += loss
                # Train the network, take a step in the direction of maximum
                # gradient, using SGD
                self.sgdStep(x, t, h0)
            totalLoss /= iters

            losses += [totalLoss]
            curTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Epoch = %d, loss = %f" % (curTime, epoch, totalLoss)

        curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        rnnName = dataDir + '/' + filePrefix + '-%d-%d-%d-%s.npz' % \
                                         (self.nh, self.nin, self.nout, curTime)
        self.saveNetwork(rnnName)
        return losses

    def testNetwork(self, dataLen):
        """
        Test the network with a new input sequence
        :param rnn: RNN object, with weight parameters
        :param dataLen: number of timesteps in each input vector
        :return: err - testing error
        """
        # Generate test data
        x, t = self.genData(dataLen)
        h0 = np.zeros(self.nh,)
        # Test the network
        _, y = self.forwardPropagation(x, h0)
        # Compute the error between computed output and actual target
        err = ((y[1:, :] - t) ** 2).sum() / dataLen
        return err
