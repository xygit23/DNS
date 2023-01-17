from DI.gcn.inits import *

import warnings
warnings.filterwarnings('ignore')
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, input, **kwargs):
        allowed_kwargs = {'name', 'logging', 'use_theta'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
            # name += '_' + str(get_layer_uid(name))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.input = input
        if isinstance(self.input, tf.SparseTensor):
            self.input_dim = self.input._my_input_dim
        else:
            self.input_dim = self.input.get_shape()[1].value

    def _call(self):
        return self.input

    def __call__(self):
        with tf.name_scope(self.name + '_cal'):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', self.input)
            outputs = self._call()
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        pass
        # for var in self.vars:
        #     tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class FullyConnected(Layer):
    """Fully Connected Layer."""

    def __init__(self, input, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        input_dim = self.input_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.name_scope(self.name):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self):
        x = self.input

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, use_theta=False, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        input_dim = self.input_dim
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.use_theta = use_theta
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.name_scope(self.name):
            if use_theta:
                self.vars['weight'] = glorot([input_dim, output_dim], name='weight')
                for i in range(len(self.support)):
                    self.vars['theta_' + str(i)] = tf.constant([1,-1,0][i], name='theta_' + str(i), dtype=tf.float32)
                    # self.vars['theta_' + str(i)] = glorot((1,1), name='theta_' + str(i))
            else:
                for i in range(len(self.support)):
                    # self.vars['weights_' + str(i)] = tf.Variable(np.ones([input_dim, output_dim], dtype=np.float32)*[1,-1,0][i], name='weights_' + str(i))
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self):
        x = self.input

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        H = None
        for i in range(len(self.support)):
            if self.use_theta:
                if H != None:
                    H = tf.sparse_add(H, self.support[i] * self.vars['theta_' + str(i)])
                else:
                    H = self.support[i] * self.vars['theta_' + str(i)]
            else:
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)],
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)

        if self.use_theta:
            output = dot(H, dot(x, self.vars['weight'], sparse=self.sparse_inputs), sparse=True)
        else:
            output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Residual(Layer):
    """Dense layer."""

    def __init__(self, input, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        input_dim = self.input_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # with tf.name_scope(self.name):
        #     self.vars['weights'] = glorot([input_dim, output_dim],
        #                                   name='weights')
        #     if self.bias:
        #         self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self):

        # # dropout
        # if self.sparse_inputs:
        #     inputs = sparse_dropout(inputs, 1-self.dropout, self.num_features_nonzero)
        # else:
        #     inputs = tf.nn.dropout(inputs, 1-self.dropout)

        bias = tf.python.ops.init_ops.zeros_initializer() if self.bias else None
        output = slim.fully_connected(self.input, self.output_dim,
                                      activation_fn=self.act,
                                      biases_initializer=bias)
        return output + self.input


class DenseNet(Layer):
    """Dense layer."""

    def __init__(self, input, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        input_dim = self.input_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.name_scope(self.name):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self):
        x = self.input

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return tf.concat([self.act(output), self.input], axis=1)

class ConvolutionDenseNet(Layer):
    """Graph convolution layer."""

    def __init__(self, input, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, use_theta=False, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        input_dim = self.input_dim
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.use_theta = use_theta
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.name_scope(self.name):
            if use_theta:
                self.vars['weight'] = glorot([input_dim, output_dim], name='weight')
                for i in range(len(self.support)):
                    self.vars['theta_' + str(i)] = tf.constant(1, name='theta_' + str(i), dtype=tf.float32)
                    # glorot((1,1), name='theta_' + str(i))
            else:
                for i in range(len(self.support)):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self):
        x = self.input

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        H = None
        for i in range(len(self.support)):
            if self.use_theta:
                if H != None:
                    H = tf.sparse_add(H, self.support[i] * self.vars['theta_' + str(i)])
                else:
                    H = self.support[i] * self.vars['theta_' + str(i)]
            else:
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)],
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)

        if self.use_theta:
            output = dot(H, dot(x, self.vars['weight'], sparse=self.sparse_inputs), sparse=True)
        else:
            output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return tf.concat([self.act(output), self.input], axis=1)
