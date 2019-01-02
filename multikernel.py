# taken from https://github.com/gparracl/MOSM

# This is a simplification and adaptation to gpflow 1.0 of the work done by Rasmus Bonnevie on issue #328, credits to him.

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.kernels import Kern
from gpflow.decors import params_as_tensors

#from gpflow._settings import settings
#float_type = settings.dtypes.float_type
#np_float_type = np.float32 if float_type is tf.float32 else np.float64

class MultiKern(Kern):
    '''this abstract kernel assumes input X where the first column is a series of integer indices and the
    remaining dimensions are unconstrained. Multikernels are designed to handle outputs from different
    Gaussian processes, specifically in the case where they are not independent and where they can be
    observed independently. This abstract class implements the functionality necessary to split
    the observations into different cases and reorder the final kernel matrix appropriately.'''
    def __init__(self, input_dim, output_dim, active_dims=None, name=None):
        Kern.__init__(self, input_dim, active_dims, name)
        self.output_dim = output_dim
   
    def subK(self, indexes, X, X2 = None):
        return NotImplementedError
    
    def subKdiag(self, indexes, X):
        return NotImplementedError

    @params_as_tensors
    def K(self, X, X2=None):
        #X, X2 = self._slice(X, X2)
        Xindex = tf.cast(X[:, 0], tf.int32) #find group indices
        Xparts, Xsplitn, Xreturn = self._splitback(X[:,1:], Xindex)
 
        if X2 is None:
            X2, X2parts, X2return, X2splitn = (X, Xparts, Xreturn, Xsplitn)
        else:
            X2index = tf.cast(X2[:, 0], tf.int32)
            X2parts, X2splitn, X2return = self._splitback(X2[:,1:], X2index)

        #construct kernel matrix for index-sorted data (stacked Xparts)
        blocks = []
        for i in range(self.output_dim):
            row_i = []
            for j in range(self.output_dim):
                row_i.append(self.subK((i, j), Xparts[i], X2parts[j]))
            blocks.append(tf.concat(row_i, 1))
        Ksort = tf.concat(blocks, 0)

        #split matrix into chunks, then stitch them together in correct order
        Ktmp = self._reconstruct(Ksort, Xsplitn, Xreturn)
        KT = self._reconstruct(tf.transpose(Ktmp), X2splitn, X2return)
        return tf.transpose(KT)

    def Kdiag(self, X):
        #X, _ = self._slice(X, None)
        Xindex = tf.cast(X[:, 0], tf.int32) #find recursion level indices
        Xparts, Xsplitn, Freturn = self._splitback(X[:,1:], Xindex)
        
        subdiags = []
        for index, data in enumerate(Xparts):
            subdiags.append(self.subKdiag(index, Xparts[index]))
        Kd = tf.concat(subdiags, 0)
        return self._reconstruct(Kd, Xsplitn, Freturn)

    def _splitback(self, data, indices):
        '''applies dynamic_partioning and calculates necessary statistics for
        the inverse mapping.'''
        parts = tf.dynamic_partition(data, indices, self.output_dim) #split data based on indices

        #to be able to invert the splitting, we need:
        splitnum = tf.stack([tf.shape(x)[0] for x in parts]) #the size of each data split
        goback = tf.dynamic_partition(tf.range(tf.shape(data)[0]), indices, self.output_dim) #indices to invert dynamic_part
        return (parts, splitnum, goback)

    def _reconstruct(self, K, splitnum, goback):
        '''uses quantities from splitback to invert a dynamic_partition'''
        tmp = tf.split(K, splitnum, axis=0)
        return tf.dynamic_stitch(goback, tmp) #stitch


