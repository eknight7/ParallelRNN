import numpy as np
import theano
import theano.tensor as TT
import random
import time
import copy
import scipy
import matplotlib.pyplot as plt

# Total Time Step
T = 10
all_dims = 1024
num_input = all_dims
num_hidden = all_dims
num_output = all_dims
batch_size = all_dims
num_input_big = 1024
num_output_big = 1024
batch_size_big = 1024

# Weight Matrices
TT_Wxh = TT.matrix(dtype='float32') # Weight from Input to Hidden
TT_Why = TT.matrix(dtype='float32') # Weight from Hidden to Output
TT_Whh = TT.matrix(dtype='float32') # Weight from Hidden to Hidden

# Input/Target Matrices
TT_x = TT.tensor3(dtype='float32') # Input matrix by time
TT_t = TT.tensor3(dtype='float32') # Target matrix by time
TT_h0 = TT.matrix(dtype='float32') # Initial hidden layer
TT_y0 = TT.matrix(dtype='float32') # Initial output layer

# Step function to link the network together
def step(TT_x_t, TT_h_tm1, TT_Wxh, TT_Whh, TT_Why):
    TT_h_t = TT.tanh(TT.dot(TT_x_t, TT_Wxh) + TT.dot(TT_h_tm1, TT_Whh))
    TT_y_t = TT.tanh(TT.dot(TT_h_t, TT_Why))
    return TT_h_t, TT_y_t

# Theano Scan Results
[TT_h, TT_y], _ = theano.scan(step, sequences=TT_x, outputs_info=[TT_h0, None], non_sequences=[TT_Wxh, TT_Whh, TT_Why])

# Calculate Gradient
error = ((TT_y - TT_t)**2).sum()
TT_Gxh, TT_Ghh, TT_Ghy = TT.grad(error, [TT_Wxh, TT_Whh, TT_Why])

# Compiled Function
fn = theano.function([TT_x,  TT_h0, TT_t, TT_Wxh, TT_Whh, TT_Why], [TT_Gxh, TT_Ghh, TT_Ghy, error])

tt_x = np.zeros((T, batch_size_big, num_input_big), dtype=np.float32)
tt_t = np.zeros((T, batch_size_big, num_output_big), dtype=np.float32)
for i in range(T):
    tt_x[i] = (scipy.misc.imread("images/X_%d.png" % (i+1)) / 255.0) - 0.5
    tt_t[i] = (scipy.misc.imread("images/T_%d.png" % (i+1)) / 255.0) - 0.5
tt_h0  = (scipy.misc.imread("images/h0.png") / 255.0) - 0.5
tt_Wxh = (scipy.misc.imread("images/Wxh.png") / 255.0) - 0.5
tt_Whh = (scipy.misc.imread("images/Whh.png") / 255.0) - 0.5
tt_Why = (scipy.misc.imread("images/Why.png") / 255.0) - 0.5
tt_x = tt_x[:,:batch_size,:num_input]
tt_t =  tt_t[:,:batch_size,:num_output]
tt_h0 = tt_h0.astype(np.float32)[:batch_size,:num_hidden]
tt_Wxh = tt_Wxh.astype(np.float32)[:num_input,:num_hidden]
tt_Whh = tt_Whh.astype(np.float32)[:num_hidden,:num_hidden]
tt_Why = tt_Why.astype(np.float32)[:num_hidden,:num_output]

learning_rate = 0.01

tt_loss_list = []
tt_time = 0
init_loss = -1
start_time = time.time()
for i in range(10):
    # Theano Version
    tt_Gxh, tt_Ghh, tt_Ghy, tt_loss = fn(tt_x,  tt_h0, tt_t, tt_Wxh, tt_Whh, tt_Why)
    tt_Wxh -= tt_Gxh * learning_rate
    tt_Whh -= tt_Ghh * learning_rate
    tt_Why -= tt_Ghy * learning_rate
    if (init_loss == -1):
        init_loss = tt_loss
    print tt_loss / (T * batch_size * num_input)
    tt_loss_list.append(tt_loss)
tt_time = time.time() - start_time
print "TT Time: %f" % (tt_time)
print "Loss %f => %f" % (init_loss / (T * batch_size * num_input), tt_loss / (T * batch_size * num_input))
