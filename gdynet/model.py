import os
import json
import pickle
import numpy as np
import tensorflow as tf
# required to make tf.compat.v1.placeholder function properly
#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.disable_v2_behavior()
from tensorflow import keras
#import keras.backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Embedding,\
    BatchNormalization, Activation, Add, concatenate, Concatenate, Permute
from .data import MDStackGenerator, MDStackGenerator_direct,\
    MDStackGenerator_vanila
from .vampnet import VampnetTools

#tf.config.run_functions_eagerly(False)

class GaussianExpand(keras.layers.Layer):
    """
    Expand distances to gaussian basis
    
    Attributes
    ----------
    num_atom : int32
        number of atoms in system
    dmin : float32
        minimum diameter for filter
    dmax : float32
        maximum diameter for filter
    step : float32
        increment for filter between min/max ranges
    var : bool
        flag for user-defined variance of gaussian

    """
    def __init__(self, num_atom, dmin, dmax, step, var=None, **kwargs):
        super(GaussianExpand, self).__init__(**kwargs)
        self.num_atom = num_atom
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.var = var if var is not None else step

    def call(self, nbr_dist):
        """
        Parameters
        ----------
        nbr_dist : float32
            gaussian distance-weighting for neighbors (shape : [B, num_atom, M])

        Returns
        -------
        bond_fea: shape (B, num_atom, M, bond_fea_len)
        """
        gfilter = tf.cast(tf.range(self.dmin, self.dmax + self.step, self.step), dtype='float32')
        return tf.exp(-(tf.cast(tf.expand_dims(nbr_dist, axis=-1), dtype='float32') - gfilter)**2 /
                      (self.var**2))
    def get_config(self):
        base_config = super().get_config()
        config = {'num_atom': self.num_atom, 'dmin': self.dmin, 'dmax': self.dmax, 'step': self.step, 'var': self.var}
        return dict(list(config.items()) + list(base_config.items()))

class PreProcessCGCNNLayer(keras.layers.Layer):
    """
    Unstacks the input atoms, neighbor lists, and target indices to
    create the bond feature tensors from nearby atoms in a graph

    Attributes
    ----------
    num_atom : int32
        number of atoms in system
    num_nbr : int32
        number of neighbors
    dmin : float32
        minimum diameter of filter
    dmax : float32
        maximum diameter of filter
    step : float32
        increment of filter between min/max ranges
    var : bool
        flag for user-defined variance of gaussian filter
    """

    def __init__(self, num_atom, num_nbr, dmin, dmax, step, var=None, **kwargs):
        super(PreProcessCGCNNLayer, self).__init__(**kwargs)
        self.num_atom = num_atom
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.N = num_atom
        self.M = num_nbr
        self.var = var if var is not None else step
    
    def batch_gather(self, atom_coords, nbr_list):
        """
        Parameters
        ----------
        atom_coords: shape (B, N, 3)
        nbr_list: shape (B, N, M)
        """
        B = tf.shape(atom_coords)[0]
        # check batch size matches for feature and indexes
        #assert B == tf.shape(nbr_list)[0]
        batch_idx = tf.cast(tf.reshape(tf.range(0, B), (B, 1, 1)), dtype='float32')
        batch_idx = tf.cast(tf.tile(batch_idx, (1, self.N, self.M)), dtype='float32')
        full_idx = tf.cast(tf.stack((batch_idx, tf.cast(nbr_list, dtype='float32')), axis=-1), dtype='int32')
        return tf.gather_nd(atom_coords, full_idx)

    def gaussian_expand(self, nbr_dist):
        """
        Parameters
        ----------
        nbr_dist : float32
            gaussian distance weighting for neighbors
        """
        gfilter = tf.cast(tf.range(self.dmin, self.dmax + self.step, self.step), dtype='float32')
        return tf.exp(-(tf.cast(tf.expand_dims(nbr_dist, axis=-1), dtype='float32') - gfilter)**2 /
                      (self.var**2))

    def pdc_dist(self,atom_coords, nbr_coords, lattice):
        """
        Distance betwewen points relative to the periodic cell

        Parameters
        ----------
        atom_coords: shape (B, N, 3)
        nbr_coords: shape (B, N, M, 3)
        lattice: shape (B, 3)
        """
        atom_coords = tf.expand_dims(atom_coords, axis=2)
        delta = tf.abs(atom_coords - nbr_coords)
        lattice = tf.expand_dims(tf.expand_dims(lattice, axis=1), axis=2)
        delta = tf.where(delta > 0.5 * lattice, delta - lattice, delta)
        return tf.sqrt(tf.reduce_sum(input_tensor=delta**2, axis=-1))
    
    def process_one(self, atom_coords, lattice, nbr_lists):
        """
        Process one graph

        Parameters
        ----------
        atom_coords : float32
            atomic coordinates
        lattice : float32
            simulation cell dimensions
        nbr_lists : int32
            neighbor lists for each atom
        """
        nbr_coords = self.batch_gather(atom_coords, nbr_lists)
        nbr_dist = self.pdc_dist(atom_coords, nbr_coords, lattice)
        bond_fea = self.gaussian_expand(nbr_dist)
        return bond_fea
    
    def call(self, inputs):
        """
        Pre-process input data

        Parameters
        ----------
        stacked_coords: shape (B, num_atom, 3, 2)
        stacked_lattices: shape (B, 3, 2)
        stacked_nbr_lists: shape (B, num_atom, M, 2)

        Returns
        -------
        nbr_lists_1: shape (B, num_atom, M)
        bond_fea_1: shape (B, num_atom, M, bond_fea_len)
        """
        stacked_coords, stacked_lattices, stacked_nbr_lists = inputs
        atom_coords_1, atom_coords_2 = tf.unstack(stacked_coords, axis=-1)
        lattice_1, lattice_2 = tf.unstack(stacked_lattices, axis=-1)
        nbr_lists_1, nbr_lists_2 = tf.unstack(stacked_nbr_lists, axis=-1)
        bond_fea_1 = self.process_one(atom_coords_1, lattice_1, nbr_lists_1)
        bond_fea_2 = self.process_one(atom_coords_2, lattice_2, nbr_lists_2)
        return [nbr_lists_1, bond_fea_1, nbr_lists_2, bond_fea_2]

    def get_config(self):
        config = {'num_atom': self.num_atom, 'dmin': self.dmin, 'dmax': self.dmax, 'step': self.step,
                'var': self.var, 'num_atom': self.N, 'num_nbr': self.M}
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))

def load_keras_optimizer(model, filepath):
    """
    Load the state of optimizer

    Parameters
    ----------
    optimizer: keras optimizer
      the optimizer for loading the state
    filepath: str
      the path of the pickle (.pkl) file that saves the optimizer state
    """
    with open(filepath, 'rb') as f:
        weight_values = pickle.load(f)
    model._make_train_function()
    model.optimizer.set_weights(weight_values)


class SaveOptimizerState(keras.callbacks.Callback):
    """
    Save the state of optimizer at the end of each epoch
    Attributes
    ----------
    filepath: str
      the path of the pickle (.pkl) file that saves the optimizer state
    """
    def __init__(self, filepath):
        super(SaveOptimizerState, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        """
        Run on the end of an epoch to log the metrics 
        Parameters
        ----------
        epoch : int32
            epoch number
        logs : int32
            keys for numbering checkpoints
        """
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = keras.backend.batch_get_value(symbolic_weights)
        with open(self.filepath, 'wb') as f:
            pickle.dump(weight_values, f)


class EpochCounter(keras.callbacks.Callback):
    """
    Count the number of epochs and save it

    Attributes
    ----------
    filepath: str
      the path of the json file that saves the current number of epochs and
      number of training stage
    train_stage: int
      current training stage, starting with 0
    """
    def __init__(self, filepath, train_stage=0):
        super(EpochCounter, self).__init__()
        self.filepath = filepath
        self.train_stage = train_stage

    def on_epoch_end(self, epoch, logs=None):
        """
        Parameters
        ----------
        epoch : int32
            epoch number
        logs : int32
            keys for numbering checkpoints
        """
        with open(self.filepath, 'w') as f:
            json.dump({'epoch': epoch,
                       'stage': self.train_stage}, f)


def reorder_predictions(raw_predictions, num_data, tau):
    """
    Reorder raw prediction array

    Parameters
    ----------
    raw_predictions: shape (num_data * (F - tau), num_atom, 2 * n_classes)
    predictions: shape (num_data, F, num_atom, n_classes)
    """
    if (raw_predictions.shape[0] % num_data != 0 or
            len(raw_predictions.shape) != 3 or
            raw_predictions.shape[2] % 2 != 0):
        raise ValueError('Bad format!')
    n_classes = raw_predictions.shape[2] // 2
    num_atom = raw_predictions.shape[1]
    raw_predictions = raw_predictions.reshape(num_data, -1, num_atom, n_classes * 2)
    assert np.allclose(raw_predictions[:, tau:, :, :n_classes], raw_predictions[:, :-tau, :, n_classes:])
    predictions = np.concatenate([raw_predictions[:, :, :, :n_classes],
                                  raw_predictions[:, -tau:, :, n_classes:]],
                                 axis=1)
    return predictions

class CGCNNLayer(keras.layers.Layer):
    """
    Concatatenate feature vectors for all atoms using lambda layer 

    Attributes
    ----------
    atom_fea_len : int32
        length of atom features
    bond_fea_len : int32
        length of bond features
    num_atom : int32
        number of atoms in system
    num_nbr : int32
        number of neighbors
    use_bn : bool
        flag for batch normalization (default True)
    """
    def __init__(self, atom_fea_len, num_atom, num_nbr, use_bn, **kwargs):
        super(CGCNNLayer, self).__init__(**kwargs)
        self.use_bn = use_bn
        # dims previously _concat_nbrs_output_shape
        self.atom_fea_len = atom_fea_len
        self.M = num_nbr 
        self.N = num_atom
        self.core = Dense(self.atom_fea_len)
        self.filter = Dense(1)
        self.bna = BatchNormalization(axis=-1)
        self.perm = Permute((1, 3, 2))
        self.softact = Activation('softmax')
        self.bnb = BatchNormalization(axis=-1)
        self.reluact = Activation('relu')
        self.finaladd = Add()
        
    def _concat_nbrs(self, atom_fea, bond_fea, nbr_list):
        """
        Concatenate neighbor features based on graph structure into a full
        feature vector. A helper function for the CGCNN Layer subclass

        B: Batch size
        N: Number of atoms in each crystal
        M: Max number of neighbors
        atom_fea_len: the length of atom features
        bond_fea_len: the length of bond features

        Parameters
        ----------
        atom_fea: (B, N, atom_fea_len)
        bond_fea: (B, N, M, bond_fea_len)
        nbr_list: (B, N, M)

        Returns
        -------
        total_fea: (B, N, M, 2 * atom_fea_len + bond_fea_len)
        """
        B = tf.shape(atom_fea)[0]
        # check batch size matches for features and indexes
        #assert B == tf.shape(nbr_list)[0]
        #assert B == tf.shape(bond_fea)[0]
        batch_idx = tf.reshape(tf.range(0, B), (B, 1, 1))
        batch_idx = tf.cast(tf.tile(batch_idx, (1, self.N, self.M)), dtype='float32')
        full_idx = tf.cast(tf.stack((batch_idx, tf.cast(nbr_list,dtype='float32')), axis=-1), dtype='int32')
        atom_nbr_fea = tf.cast(tf.gather_nd(atom_fea, full_idx), dtype='float32')
        atom_fea = tf.cast(tf.tile(tf.expand_dims(atom_fea, 2), [1, 1, self.M, 1]), dtype='float32')
        full_fea = tf.concat([atom_fea, atom_nbr_fea, bond_fea], axis=-1)
        return full_fea

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            list of layer inputs
        """
        atom_fea, bond_fea, nbr_list = inputs
        total_fea = self._concat_nbrs(atom_fea, bond_fea, nbr_list)
        # total_fea shape (None, N, M, 2 * atom_fea_len + bond_fea_len)
        nbr_core = self.core(total_fea)
        nbr_filter = self.filter(total_fea)
        if self.use_bn:
            nbr_core = self.bna(nbr_core)
        nbr_filter = self.perm(nbr_filter)
        nbr_filter = self.softact(nbr_filter)
        nbr_filter = self.perm(nbr_filter)
        # nbr_filter = keras.activations.softmax(nbr_filter, axis=-2)
        nbr_core = self.reluact(nbr_core)
        nbr_sumed = Lambda(lambda x : tf.reduce_mean(x[0] * x[1], axis=2))([nbr_filter, nbr_core])
        if self.use_bn:
            nbr_sumed = self.bnb(nbr_sumed)
        addition = self.finaladd([atom_fea, nbr_sumed])
        out = self.reluact(addition)
        return out

    def get_config(self): 
        config = {'atom_fea_len': self.atom_fea_len, 'use_bn': self.use_bn, 'num_atom': self.N, 'num_nbr': self.M}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CrystalFeaturePoolingLayer(keras.layers.Layer):
    """CrystalFeaturePoolingLayer
    Pool the local crystal features for the target atoms
    (used in build_cgcnn_model within GDyNet class)

    Attributes
    ----------
    num_target : int32
        number of target atoms
    """
    def __init__(self, num_target, **kwargs):
        super(CrystalFeaturePoolingLayer, self).__init__(**kwargs)
        self.N0 = num_target

    def call(self, inputs):
        """
        Pool the atom_fea of target atoms together using their indexes.

        Parameters
        ----------
        atom_fea: (B, N, atom_fea_len)
        target_index: (B, N0)

        N: number of atoms in each batch.
        N0: number of target atoms in each batch.

        Note here we assume the target atoms are single atoms (e.g. Li-ions).
        TODO: implement a pooling function for cases where the target atoms are
        molecules (e.g. H2O).

        Returns
        -------
        crys_fea: (B, N0, atom_fea_len)
        """
        atom_fea, target_index_inp = inputs
        B = tf.shape(atom_fea)[0]
        # verify batch size agreement
        #assert B == tf.shape(target_index_inp)[0]
        batch_idx = tf.cast(tf.reshape(tf.range(0, B), (B, 1)), dtype='float32')
        batch_idx = tf.cast(tf.tile(batch_idx, (1, self.N0)), dtype='float32')
        full_idx = tf.cast(tf.stack((batch_idx, tf.cast(target_index_inp, dtype='float32')), axis=-1), dtype='int32')
        return tf.gather_nd(atom_fea, full_idx)


    def get_config(self):
        base_config = super().get_config()
        config = {'num_target': self.N0}
        return dict(list(config.items()) + list(base_config.items()))

class CGCNNModel(keras.Model):
    """CGCNNModel
    Crystal graph convolutional neural network for featurization of
    atomic neighbor lists around target atoms. Used as part of GDyNet

    Attributes
    ----------
    atom_fea_len : int32
        length of atom features
    n_conv : int32
        number of CGCNN layers in lobe/sub-model
    num_atom : int32
        number of atoms in system
    num_target : int32
        number of target atoms
    num_nbr : int32
        number of neighbors
    bond_fea_len : int32
        length of bond features
    use_bn : bool
        flag for batch normalization (default = True)
    """
    def __init__(self, atom_fea_len, n_conv, num_atom, num_target, num_nbr, use_bn, **kwargs):
        super(CGCNNModel, self).__init__(**kwargs)
        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv
        self.num_atom = num_atom
        self.num_target = num_target
        self.use_bn = use_bn
        self.num_nbr = num_nbr
       # build layers 
        self.atom_fea_embedding = Embedding(input_dim=100, output_dim=self.atom_fea_len)
        # why the fuck is this here
        # bond_fea, nbr_list = self.bond_fea_inp, self.nbr_list_inp
        self.cgcnn_layers = []
        for _ in range(self.n_conv):
            self.cgcnn_layers.append(CGCNNLayer(atom_fea_len=self.atom_fea_len, num_atom=self.num_atom, num_nbr=self.num_nbr, use_bn=self.use_bn))
        self.crys_fea_pool = CrystalFeaturePoolingLayer(num_target=self.num_target)
        self.reluact = Activation('relu')
        self.dense = Dense(self.atom_fea_len)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            list of inputs
        """
        atom_types_inp, bond_fea_inp, nbr_list_inp, target_index_inp = inputs
        x = self.atom_fea_embedding(atom_types_inp)
        # convolutions
        for layer in self.cgcnn_layers:
            x = layer([x, bond_fea_inp, nbr_list_inp])
        x = self.crys_fea_pool([x, target_index_inp])
        x = self.reluact(x)
        x = self.dense(x)
        x = self.reluact(x)
        return x

    def get_config(self):
        config = {'num_atom': self.num_atom, 'atom_fea_len': self.atom_fea_len, 'n_conv': self.n_conv, 'num_target': self.num_target, 
                'use_bn': self.use_bn, 'num_nbr': self.num_nbr}
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))



class GDyNetModel(keras.Model):
    """GDyNetModel
    Graph dynamical network model for deep learning of particle dynamics
    using timed trajectory data
    Attributes
    ----------
    mode : str
        method of graph preprocessing (kdtree, direct, or vanilla)
    atom_fea_len : int32
        length of atom features
    n_conv : int32
        number of CGCNN layers in lobe/sub-model
    dmin : float32
        minimum diameter of filter
    dmax : float32
        maximum diameter of filter
    step : float32
        increment of filter between min/max range
    num_atom : int32
        number of atoms in system
    num_target : int32
        number of target atoms
    num_nbr : int32
        number of neighbors
    use_bn : bool
        flag for batch normalization (default = True)
    n_classes :
        n_classes
    """
    def __init__(self, mode, atom_fea_len, n_conv, dmin, dmax, step, num_atom, num_target, num_nbr, use_bn, n_classes, **kwargs):
        super(GDyNetModel, self).__init__(**kwargs)
        if mode not in ['kdtree', 'direct', 'vanilla']: 
            raise ValueError('`mode` must in `kdtree`, `direct`, or `vanilla`')
        # Model variables
        self.mode = mode
        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv 
        # pre-processing variables
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.bond_fea_len = len(np.arange(self.dmin, self.dmax + self.step, self.step))
        self.num_atom = num_atom
        self.num_target = num_target
        self.num_nbr = num_nbr
        self.use_bn = use_bn
        # used by vanilla build
        self.n_classes = n_classes

        if mode == 'kdtree':
            """build gdynet using graphs constructed using the `kdtree` backend"""
            self.preprocess = PreProcessCGCNNLayer(num_atom=self.num_atom, dmin=self.dmin, dmax=self.dmax, step=self.step,
                    num_nbr=self.num_nbr)

            self.branch_1 = CGCNNModel(atom_fea_len=self.atom_fea_len, n_conv=self.n_conv, num_atom=self.num_atom,
                    num_target=self.num_target, num_nbr=self.num_nbr, use_bn=self.use_bn)

            self.branch_2 = CGCNNModel(atom_fea_len=self.atom_fea_len, n_conv=self.n_conv, num_atom=self.num_atom,
                    num_target=self.num_target, num_nbr=self.num_nbr, use_bn=self.use_bn)

        elif mode == 'direct':
            """build gdynet with graphs constructed using the `direct` backend"""
            self.gaussian_expand_1 = GaussianExpand(dmax=self.dmax, dmin=self.dmin, step=self.step, num_atom=self.num_atom)
            self.gaussian_expand_2 = GaussianExpand(dmax=self.dmax, dmin=self.dmin, step=self.step, num_atom=self.num_atom)
            self.branch_1 = CGCNNModel(atom_fea_len=self.atom_fea_len, n_conv=self.n_conv, num_atom=self.num_atom,
                    num_target=self.num_target, num_nbr=self.num_nbr, use_bn=self.use_bn)
            self.branch_2 = CGCNNModel(atom_fea_len=self.atom_fea_len, n_conv=self.n_conv, num_atom=self.num_atom,
                    num_target=self.num_target, num_nbr=self.num_nbr, use_bn=self.use_bn)

        elif mode =='vanilla':
            """build a vanilla VAMPnet as baseline"""
            self.bn_layer = BatchNormalization()
            self.dense_layers = [Dense(self.atom_fea_len, activation='relu')
                            for _ in range(self.n_conv)]
            self.softmax = Dense(self.n_classes, activation='softmax')

        self.merge = Concatenate()
    
    #@tf.function
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            list of tensor inputs
        """
        if self.mode =='kdtree':
            stacked_coords_inp, stacked_lattices_inp, stacked_nbr_lists_inp, atom_types_inp, target_index_inp = inputs
            nbr_lists_1, bond_fea_1, nbr_lists_2, bond_fea_2 = self.preprocess([stacked_coords_inp, stacked_lattices_inp, stacked_nbr_lists_inp])
            x1 = self.branch_1([atom_types_inp, bond_fea_1, nbr_lists_1, target_index_inp])
            x2 = self.branch_2([atom_types_inp, bond_fea_2, nbr_lists_2, target_index_inp])
            merged = self.merge([x1, x2])

        elif self.mode == 'direct':
            atom_types_inp, target_index_inp, bond_dist_1_inp, bond_dist_2_inp, nbr_lists_1, nbr_lists_2 = inputs
            bond_fea_1 = self.gaussian_expand_1(bond_dist_1_inp)
            bond_fea_2 = self.gaussian_expand_2(bond_dist_2_inp)
            x1 = self.branch_1([atom_types_inp, bond_fea_1, nbr_lists_1, target_index_inp])
            x2 = self.branch_2([atom_types_inp, bond_fea_2, nbr_lists_2, target_index_inp])
            merged = self.merge([x1, x2])

        elif self.mode == 'vanilla':
            traj_coords_1_inp, traj_coords_2_inp = inputs
            x1 = self.bn_layer(traj_coords_1_inp)
            x2 = self.bn_layer(traj_coords_2_inp)
            for layer in self.dense_layers:
                x1 = layer(traj_coords_1_inp)
                x2 = layer(traj_coords_2_inp)

            x1 = self.softmax(x1)
            x2 = self.softmax(x2)
            merged = self.merge([x1, x2])

        return merged

    #def compute_output_shape(self, input_shape):
    #    return (input_shape[0], self.num_atom, self.n_classes)

    def get_config(self):
        config = {'mode': self.mode, 'atom_fea_len': self.atom_fea_len, 'bond_fea_len': self.bond_fea_len, 'n_conv': self.n_conv, 'dmin': self.dmin,
                'dmax':self.dmax, 'step':self.step, 'num_atom': self.num_atom, 'num_target': self.num_target,
                'num_nbr': self.num_nbr, 'use_bn': self.use_bn, 'n_classes': self.n_classes}
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))

class VAMPOneMetric(keras.metrics.Metric):
    def __init__(self, name='VAMP_1_scores', **kwargs):
        super(VAMPOneMetric, self).__init__(name=name, **kwargs)
        self.vamp_tools = VampnetTools()
        self.vamp_one = self.add_weight(name='vamp_one', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.vamp_tools.metric_VAMP(y_true, y_pred)
        if sample_weight is not None:
            values = tf.multiply(values, sample_weight)
        self.vamp_one.assign(tf.math.reduce_mean(values))

    def result(self):
        return self.vamp_one

    def reset_state(self):
        self.vamp_one.assign(0.0)

    def get_config(self):
        base_config = super().get_config()
        config = {'vamp_tools' : self.vamp_tools}
        return dict(list(base_config.items()) + list(config.items()))

class VAMPTwoMetric(keras.metrics.Metric):
    def __init__(self, name='VAMP_2_scores', **kwargs):
        super(VAMPTwoMetric, self).__init__(name=name, **kwargs)
        self.vamp_tools = VampnetTools()
        self.vamp_two = self.add_weight(name='vamp_two', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.vamp_tools.metric_VAMP2(y_true, y_pred)
        if sample_weight is not None:
            values = tf.multiply(values, sample_weight)
        self.vamp_two.assign(tf.math.reduce_mean(values))

    def result(self):
        return self.vamp_two

    def reset_state(self):
        self.vamp_two.assign(0.0)

    def get_config(self):
        base_config = super().get_config()
        config = {'vamp_tools' : self.vamp_tools}
        return dict(list(config.items()) + list(base_config.items()))

class VAMPOneLoss(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='VAMP_1_loss', **kwargs):
        super(VAMPOneLoss, self).__init__(name=name, **kwargs)
        self.vamp_tools = VampnetTools()
    def call(self, y_true, y_pred):
        return self.vamp_tools._loss_VAMP_sym(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        config = {'vamp_tools': self.vamp_tools}
        return dict(list(base_config.items()) + list(config.items()) + list(config.items()))


class VAMPTwoLoss(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='VAMP_2_loss', **kwargs):
        super(VAMPTwoLoss, self).__init__(name=name, **kwargs)
        self.vamp_tools = VampnetTools()
    def call(self, y_true,y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        return self.vamp_tools.loss_VAMP2_autograd(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        config = {'vamp_tools' : self.vamp_tools}
        return dict(list(base_config.items()) + list(config.items()))


class GDyNet(object):
    """
    Graph Dynamical Network with VAMP loss to analyze time series data.

    Attributes
    ----------
    train_flist : str
        training graphs file name in job directory
    val_flist : str
        validation graphs file name in job directory
    test_flist : str
        testing graphs file name in job directory
    job_dir : str
        job directory for I/O
    mode : str
        method of graph preprocessing (kdtree, direct, vanilla)
    tau : float32
        Markov state model tau (time lag hyperparameter)
    n_classes : int32
        number of classes in Markov state model
    k_eig : int32
        k_eig
    no_pool : bool
        flag to disable pooling (default False)
    atom_fea_len : int32
        length of atom features
    n_conv : int32 
        number of CGCNN layers in lobes/sub-models
    train_n_li :
        train_n_li
    val_n_li :
        val_n_li
    test_n_li :
        test_n_li
    dmin : float32
        minimum diameter of gaussian filter
    dmax : float32
        maximum diameter of gaussian filter
    step : float32
        increment of gaussian filter
    learning_rate : float32
        optimizer learning rate
    batch_size : int32
        batch size
    use_bn : bool
        flag for batch normalization (default = True)
    n_epoch : int32
        number of training epochs
    shuffle : bool
        flag to enable shuffling (default = True)
    random_seed : int32
        random seed
    transfer : bool
        flag to enable transfer training on new data sample for same system
    prev_data_dir : str
        path to last model for reloading and transfer learning
    """
    def __init__(self, train_flist, val_flist, test_flist,
                 job_dir='./', mode='direct',
                 tau=1, n_classes=2, k_eig=0, no_pool=False,
                 atom_fea_len=16, n_conv=3,
                 train_n_li=None, val_n_li=None, test_n_li=None,
                 dmin=0., dmax=7., step=0.2,
                 learning_rate=0.0005, batch_size=16, use_bn=True,
                 n_epoch=10, shuffle=True, random_seed=123,
                 transfer=False, prev_data_dir=None):
        if mode not in ['kdtree', 'direct', 'vanilla']:
            raise ValueError('`mode` must in `kdtree`, `direct`, or `vanilla`')
        self.train_flist = train_flist
        self.val_flist = val_flist
        self.test_flist = test_flist
        self.job_dir = job_dir
        self.mode = mode
        self.tau = tau
        self.n_classes = n_classes
        self.k_eig = k_eig
        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_bn = use_bn
        self.n_epoch = n_epoch
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.transfer = transfer
        self.prev_data_dir=None
        self.vamp = VampnetTools(epsilon=1e-5, k_eig=self.k_eig)
        #self.metrics = [VAMPOneMetric(), VAMPTwoMetric()]
        self.losses = [self.vamp.loss_VAMP2_autograd, self.vamp._loss_VAMP_sym, self.vamp.loss_VAMP2_autograd]

        try:
            if self.mode == 'kdtree':
                self.load_data()
            elif self.mode == 'direct':
                self.load_data_direct()
            elif self.mode == 'vanilla':
                self.load_data_vanilla()
        except ValueError:
            print(f"mode {self.mode} for preprocessing is not kdtree/direct/vanilla")

    def build(self):
        """
        Assemble inputs and layer architecture of the GDyNet model.
        Builds with a certain neighbor featurization algorithm (kdtree, direct, etc.)
        Input sizes are written here for user access in self.inputs, but the
        generators determine the fixed values for num_atom, num_nbr, etc.
        """
        try:
            if self.mode == 'kdtree':
                """build gdynet with graphs constructed using the `kdtree` backend"""
                # form model inputs 
                self.stacked_coords_inp = Input(shape=(self.num_atom, 3, 2), dtype='float32')
                self.stacked_lattices_inp = Input(shape=(3, 2), dtype='float32')
                self.stacked_nbr_lists_inp = Input(shape=(self.num_atom, self.num_nbr, 2), dtype='float32')
                self.atom_types_inp = Input(shape=(self.num_atom, ), dtype='float32')
                self.target_index_inp = Input(shape=(self.num_target,), dtype='float32')

                self.input_list = [self.stacked_coords_inp, self.stacked_lattices_inp, self.stacked_nbr_lists_inp]

            elif self.mode == 'direct': 
                """build gdynet with graphs constructed using the `direct` backend"""
                # form model inputs, are these even used now that a generator is employed?
                self.atom_types_inp = Input(shape=(self.num_atom, ), dtype='float32')
                self.target_index_inp = Input(shape=(self.num_target, ), dtype='float32')
                self.bond_dist_1_inp = Input(shape=(self.num_atom, self.num_nbr), dtype='float32')
                self.bond_dist_2_inp = Input(shape=(self.num_atom, self.num_nbr), dtype='float32')
                self.nbr_list_1_inp = Input(shape=(self.num_atom, self.num_nbr), dtype='float32')
                self.nbr_list_2_inp = Input(shape=(self.num_atom, self.num_nbr), dtype='float32')
                self.input_list = [self.atom_types_inp, self.target_index_inp, self.bond_dist_1_inp, self.bond_dist_2_inp, self.nbr_list_1_inp, self.nbr_list_2_inp]

            elif self.mode == 'vanilla':
                """build a vanilla VAMPnet as baseline"""
                self.traj_coords_1_inp = Input(shape=(self.num_atom, 3))
                self.traj_coords_2_inp = Input(shape=(self.num_atom, 3))
                self.input_list = [self.traj_coords_1_inp, self.traj_coords_2_inp]
        except ValueError:
            print('Mode must be kdtree/direct/vanilla')
        # construct Model state and print configuration
        if self.transfer == False:
            self.model = GDyNetModel(mode=self.mode, atom_fea_len=self.atom_fea_len, n_conv=self.n_conv, dmin=self.dmin, dmax=self.dmax,
                step=self.step, num_atom=self.num_atom, num_target=self.num_target, num_nbr=self.num_nbr,
                use_bn=self.use_bn, n_classes=self.n_classes)
        else:
           self.model = tf.keras.load_models('first_model', custom_objects = {'loss_VAMP2_autograd': self.vamp.loss_VAMP2_autograd,
                                                                            '_loss_VAMP_sym': self.vamp._loss_VAMP_sym,
                                                                            'metric_VAMP': self.vamp.metric_VAMP,
                                                                            'metric_VAMP2': self.vamp.metric_VAMP2,
                                                                            'PreProcessCGCNNLayer': PreProcessCGCNNLayer,
                                                                            'GaussianExpand': GaussianExpand,
                                                                            'CrystalFeaturePoolingLayer': CrystalFeaturePoolingLayer,
                                                                            'CGCNNLayer': CGCNNLayer,
                                                                            'CGCNNModel': CGCNNModel,
                                                                            'reorder_predictions': reorder_predictions,
                                                                            'load_keras_optimizer': load_keras_optimizer
                                                                            })
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)


    def train_model(self):
        """
        Train after compiling the model, setting keras callbacks,
        and checking for an old model to continue from (is_continue flag)
        """
        init_epoch, init_stage, self.is_continue = self.check_train_state()
        for l_index, loss_function in enumerate(self.losses):
            # skip this round if previous training records are found
            if l_index < init_stage:
                continue
            self.model.compile(optimizer=self.optimizer,
                               loss=loss_function,
                               metrics=[self.vamp.metric_VAMP,
                                   self.vamp.metric_VAMP2])
            if self.is_continue:
                opti_state_file = os.path.join(self.job_dir, 'opti_state.pkl')
                load_keras_optimizer(self.model, opti_state_file)
                self.is_continue = False
            self.best_model_path = os.path.join(
                self.job_dir, 'best_model_{}'.format(l_index))
            self.last_model_path = os.path.join(
                self.job_dir, 'last_model'.format(l_index))
            self.train_logger_path = os.path.join(
                self.job_dir, 'train_{}.log'.format(l_index))
            self.opti_state_path = os.path.join(
                self.job_dir, 'opti_state.pkl')
            self.train_state_path = os.path.join(
                self.job_dir, 'train_state.json')
            self.callbacks = [keras.callbacks.TerminateOnNaN(),
                         keras.callbacks.ModelCheckpoint(
                         self.best_model_path,
                         monitor='val_metric_VAMP2',
                         save_best_only=True, save_weights_only=False,
                         mode='max', options=tf.saved_model.SaveOptions(experimental_io_device='CPU:0')),
                         keras.callbacks.ModelCheckpoint(
                         self.last_model_path,
                         save_weights_only=False, options=tf.saved_model.SaveOptions(experimental_io_device='CPU:0')),
                         keras.callbacks.CSVLogger(
                         self.train_logger_path,
                         separator=',', append=True),
                         EpochCounter(self.train_state_path, train_stage=l_index),
                         SaveOptimizerState(self.opti_state_path)]
            if self.is_continue:
                weights_file = os.path.join(self.job_dir, 'last_model'.format(init_stage))
                self.reload_model(weights_file)
            else:
                self.model.fit(x=self.train_generator,
                        validation_data=self.val_generator,
                        use_multiprocessing=False,
                        epochs=self.n_epoch,
                        callbacks=self.callbacks,
                        initial_epoch=init_epoch)
        # these help save memory between training phases (VAMP2-VAMP1-VAMP2)
        del self.train_generator
        del self.val_generator
        del self.model
        return None

    def load_data(self):
        """Load data with kdtree backend
        """
        self.train_generator = MDStackGenerator(self.train_flist,
                                                tau=self.tau,
                                                batch_size=self.batch_size,
                                                random_seed=self.random_seed,
                                                shuffle=self.shuffle)
        self.val_generator = MDStackGenerator(self.val_flist,
                                              tau=self.tau,
                                              batch_size=self.batch_size,
                                              random_seed=self.random_seed,
                                              shuffle=self.shuffle)
        self.num_atom = self.train_generator[0][0][0].shape[1]
        self.num_nbr = self.train_generator[0][0][2].shape[2]
        self.num_target = self.train_generator[0][0][4].shape[1]

    def load_data_direct(self):
        """Load data with direct backend
        """
        self.train_generator = MDStackGenerator_direct(
            self.train_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.val_generator = MDStackGenerator_direct(
            self.val_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.num_atom = self.train_generator[0][0][0].shape[1]
        self.num_nbr = self.train_generator[0][0][2].shape[2]
        self.num_target = self.train_generator[0][0][1].shape[1]

    def load_data_vanilla(self):
        """Load data with vanilla backend
        """
        self.train_generator = MDStackGenerator_vanila(
            self.train_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.val_generator = MDStackGenerator_vanila(
            self.val_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.num_atom = self.train_generator[0][0][0].shape[1]

    def check_train_state(self):
        """Check if previous runs have found an optimum state to use as a
        starting point for this run (stored as json flags)
        """ 
        try:
            with open(os.path.join(self.job_dir, 'train_state.json')) as f:
                data = json.load(f)
                init_epoch, init_stage = data['epoch'], data['stage']
            if init_epoch == self.n_epoch - 1:
                init_epoch = 0
                init_stage += 1
            else:
                init_epoch += 1
            self.is_continue = True
        except IOError:
            init_epoch, init_stage, self.is_continue = 0, 0, False
        return init_epoch, init_stage, self.is_continue
        
    def reload_model(self, weights_file):
        """Reload model from a previously saved SavedModel folder
        """
        try:
            self.model = tf.keras.models.load_model(weights_file, custom_objects = {'loss_VAMP2_autograd': self.vamp.loss_VAMP2_autograd,
                                                                                '_loss_VAMP_sym': self.vamp._loss_VAMP_sym,
                                                                                'metric_VAMP': self.vamp.metric_VAMP,
                                                                                'metric_VAMP2': self.vamp.metric_VAMP2,
                                                                                'CGCNNModel': CGCNNModel,
                                                                                'PreProcessCGCNNLayer': PreProcessCGCNNLayer,
                                                                                'GaussianExpand': GaussianExpand,
                                                                                'CrystalFeaturePoolingLayer': CrystalFeaturePoolingLayer,
                                                                                'reorder_predictions': reorder_predictions,
                                                                                'load_keras_optimizer': load_keras_optimizer})
        except tf.errors.FailedPreconditionError:
            print('Model was not reloaded successfully')
        print('Model reloaded successfully')
        return None


    def test_predict(self, result_dir):
        """Test predictions and generate output for training/MSM results
        Parameters
        ----------
        result_dir : str
            results directory
        """
        if self.mode == 'kdtree':
            self.test_generator = MDStackGenerator(
                self.test_flist,
                tau=self.tau,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                shuffle=False)
            self.num_atom = self.test_generator[0][0][0].shape[1]
            self.num_nbr = self.test_generator[0][0][2].shape[2]
            self.num_target = self.test_generator[0][0][4].shape[1] 
            self.build()

        elif self.mode == 'direct':
            self.test_generator = MDStackGenerator_direct(
                self.test_flist,
                tau=self.tau,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                shuffle=False)
            self.num_atom = self.test_generator[0][0][0].shape[1]
            self.num_nbr = self.test_generator[0][0][2].shape[2]
            self.num_target = self.test_generator[0][0][1].shape[1]
            self.build()

        elif self.mode == 'vanilla':
            self.test_generator = MDStackGenerator_vanila(
                self.test_flist,
                tau=self.tau,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                shuffle=False)
            self.num_atom = self.train_generator[0][0][0].shape[1]
            self.build()

        else:
            raise ValueError
        # load weights
        weights_file = os.path.join(self.job_dir, 'last_model')
        self.build()

        self.reload_model(weights_file)
        raw_preds = self.model.predict(x=self.test_generator, use_multiprocessing=False)
        preds = reorder_predictions(raw_preds, len(self.test_flist),
                                    self.tau)
        np.save(os.path.join(result_dir, 'test_pred.npy'), preds)
        # shape is read from raw_preds not the placeholder fanymore
        #preds_placeholder = Input(name='preds_placeholder', shape= (len(self.test_flist), -1, self.num_atom, self.n_classes), dtype='float32')
        metric_vamp = self.vamp.metric_VAMP(None, preds)
        metric_vamp2 = self.vamp.metric_VAMP2(None, preds)
        results = [metric_vamp, metric_vamp2]
        np.savetxt(os.path.join(result_dir, 'test_eval.csv'), np.array(results), delimiter=',')
        self.model.save(self.last_model_path, include_optimizer = True, save_format='tf',options=tf.saved_model.SaveOptions(experimental_io_device='CPU:0'))
