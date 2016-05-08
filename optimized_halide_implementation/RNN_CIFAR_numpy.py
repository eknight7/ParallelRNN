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

def trainNetwork(np_x,  np_h0, np_t, np_Wxh, np_Whh, np_Why):
    # Setup for Forward propagation
    np_h = np.zeros((T+1, batch_size, num_hidden), dtype=np.float32)
    np_y = np.zeros((T+1, batch_size, num_output), dtype=np.float32)
    np_h[0] = np_h0
    np_x = np.append(np.zeros((1, batch_size, num_input), dtype=np.float32), np_x, axis=0)
    np_t = np.append(np.zeros((1, batch_size, num_output), dtype=np.float32), np_t, axis=0)
    # Compute Hidden
    for i in range(1, T+1):
        np_h[i] = np.tanh(np.dot(np_x[i], np_Wxh) + np.dot(np_h[i-1], np_Whh))
        np_y[i] = np.tanh(np.dot(np_h[i], np_Why))
    # Setup for Back propagation
    np_Gxh = np.zeros(np_Wxh.shape, dtype=np.float32)
    np_Ghh = np.zeros(np_Whh.shape, dtype=np.float32)
    np_Ghy = np.zeros(np_Why.shape, dtype=np.float32)

    # Back propagation
    dEdy_before_tanh = 2 * (np_y - np_t) * (1 - (np_y ** 2))
    dEdh_before_tanh = np.zeros((batch_size, num_hidden), dtype=np.float32)
    for i in range(T, 0, -1):
        dEdh_before_tanh = (np.dot(dEdy_before_tanh[i], np_Why.T) + np.dot(dEdh_before_tanh, np_Whh.T)) * (1 - (np_h[i] ** 2))
        np_Ghy += np.dot(np_h[i].T, dEdy_before_tanh[i]) # Update Gradient for batch
        np_Ghh += np.dot(np_h[i-1].T, dEdh_before_tanh) # Update Gradient for batch
        np_Gxh += np.dot(np_x[i].T, dEdh_before_tanh) # Update Gradient for batch
    loss = np.sum((np_y - np_t) ** 2)
    return np_Gxh, np_Ghh, np_Ghy, loss

def testNetwork(np_x,  np_h0, np_t, np_Wxh, np_Whh, np_Why):
    # Setup for Forward propagation
    np_h = np.zeros((T+1, batch_size, num_hidden), dtype=np.float32)
    np_y = np.zeros((T+1, batch_size, num_output), dtype=np.float32)
    np_h[0] = np_h0
    np_x = np.append(np.zeros((1, batch_size, num_input)), np_x, axis=0)
    np_t = np.append(np.zeros((1, batch_size, num_output)), np_t, axis=0)
    # Compute Hidden
    for i in range(1, T+1):
        np_h[i] = np.tanh(np.dot(np_x[i], np_Wxh) + np.dot(np_h[i-1], np_Whh))
        np_y[i] = np.tanh(np.dot(np_h[i], np_Why))
    return np_y

np_x = np.zeros((T, batch_size_big, num_input_big), dtype=np.float32)
np_t = np.zeros((T, batch_size_big, num_output_big), dtype=np.float32)
for i in range(T):
    np_x[i] = (scipy.misc.imread("images/X_%d.png" % (i+1)) / 255.0) - 0.5
    np_t[i] = (scipy.misc.imread("images/T_%d.png" % (i+1)) / 255.0) - 0.5
np_h0  = (scipy.misc.imread("images/h0.png") / 255.0) - 0.5
np_Wxh = (scipy.misc.imread("images/Wxh.png") / 255.0) - 0.5
np_Whh = (scipy.misc.imread("images/Whh.png") / 255.0) - 0.5
np_Why = (scipy.misc.imread("images/Why.png") / 255.0) - 0.5
np_x = np_x[:,:batch_size,:num_input]
np_t = np_t[:,:batch_size,:num_output]
np_h0 = np_h0.astype(np.float32)[:batch_size,:num_hidden]
np_Wxh = np_Wxh.astype(np.float32)[:num_input,:num_hidden]
np_Whh = np_Whh.astype(np.float32)[:num_hidden,:num_hidden]
np_Why = np_Why.astype(np.float32)[:num_hidden,:num_output]

learning_rate = 0.01

np_loss_list = []
np_time = 0
init_loss = -1
start_time = time.time()
for i in range(10):
    # Numpy Version
    np_Gxh, np_Ghh, np_Ghy, np_loss = trainNetwork(np_x,  np_h0, np_t, np_Wxh, np_Whh, np_Why)
    np_Wxh -= np_Gxh * learning_rate
    np_Whh -= np_Ghh * learning_rate
    np_Why -= np_Ghy * learning_rate
    if (init_loss == -1):
        init_loss = np_loss
    np_loss_list.append(np_loss)
    print np_loss / (T * batch_size * num_input)
np_time = time.time() - start_time

print "NP Time: ", np_time
print "Loss %f => %f" % (init_loss / (T * batch_size * num_input), np_loss / (T * batch_size * num_input))

