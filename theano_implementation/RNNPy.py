__author__ = 'Esha Uboweja'

import numpy as np
from datetime import datetime

class RNNPy(object):

    def __init__(self, nh, nin, nout):
        """
        RNN framework setup
        :param nh: Number of hidden units
        :param nin: Number of input units
        :param nout: Number of output units
        """
        # Number of hidden units
        self.nh = nh
        # Number of input units
        self.nin = nin
        # Number of output units
        self.nout = nout
