# RNN implementation in Theano
# This RNN learns a 1-D  vertical edge filter

import numpy as np
import theano
import theano.tensor as TT
import random
from datetime import datetime

# Number of hidden units
nh = 4
# Number of input units
nin = 1
# Number of output units
nout = 1

# Input (first dim is time)
x = TT.matrix()
# Target (first dim is time)
t = TT.matrix()
# Initial hidden state of RNNs
h0 = TT.vector()
# Learning rate
lr = TT.scalar()
# Recurrent hidden node weights
W_hh = theano.shared(np.random.uniform(size=(nh, nh), low=-0.01, high=0.01))
# Input layer to hidden layer weights
W_xh = theano.shared(np.random.uniform(size=(nin, nh), low=-0.01, high=0.01))
# Hidden layer to output layer weights
W_hy = theano.shared(np.random.uniform(size=(nh, nout), low=-0.01, high=0.01))

# Forward step function (recurrent)
# Nonlinear activation function - one of tanh, sigmoid, ReLU
# x_t - input at timestep t
# h_tm1 - hidden state at timestep (t-1)
# Result - Update hidden state at timestep t, h_t, update output at timestep t
def step(x_t, h_tm1, W_hh, W_xh, W_hy):
  h_t = TT.tanh(TT.dot(x_t, W_xh) + TT.dot(h_tm1, W_hh))
  y_t = TT.dot(h_t, W_hy)
  return h_t, y_t

# Compute the hidden state and output for the entire input sequence
# (first dim is time)
[h, y], _ = theano.scan(step,
                        sequences=x, 
                        outputs_info=[h0, None], 
                        non_sequences=[W_hh, W_xh, W_hy])

# Error between output and target
error = ((y - t) ** 2).sum()
# BPTT
# Gradients
gW_hh, gW_xh, gW_hy = TT.grad(error, [W_hh, W_xh, W_hy])
# Training function
train_fn = theano.function([h0, x, t, lr],
                           [error, y],
                           updates={W_hh: W_hh - lr * gW_hh,
                                    W_xh: W_xh - lr * gW_xh,
                                    W_hy: W_hy - lr * gW_hy})

# Generate data
def getData(dataLen):
  x = np.reshape(np.asarray([random.random() for v in xrange(dataLen)]), 
                (dataLen, 1))
  t = [x[j + 2] - x[j] for j in xrange(dataLen - 2)]
  t = np.reshape(np.asarray([0, 0] + t), (dataLen, 1))
  return x, t

# Network training
def trainNetwork(iters, dataLen, step):
  for i in xrange(iters):
    # Generate training data
    x, t = getData(dataLen)
    # Train the network
    err, y = train_fn(np.zeros(nh,), x, t, 0.001)

# Save the network weights learnt after training
def saveNetwork(W_hh, W_xh, W_hy):
  curTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  W_hh, W_xh, W_hy = W_hh.get_value(), W_xh.get_value(), W_hy.get_value()
  np.savez('./network_params/rnn_1d_verticall-%d-%d-%d-%s.npz' \
            % (nh, nin, nout, curTime), W_hh=W_hh, W_xh=W_xh, W_hy=W_hy)

# Network testing
def testNetwork(dataLen):
  # Generate test data
  x, t = getData(dataLen)
  # Test the network
  _, y = train_fn(np.zeros(nh,), x, np.zeros((dataLen, 1)), 0)
  # Compute error between computed output and actual target
  err = ((y - t) ** 2).sum() / dataLen
  return err

iters = 1000000
xN = 10
step = 0.001

# Train the network
trainNetwork(iters, xN, step)

# Save the network
saveNetwork(W_hh, W_xh, W_hy)

# Test the network
tN = 10000
print "Test error: ", testNetwork(tN)
