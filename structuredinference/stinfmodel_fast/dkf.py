from __future__ import division
import six.moves.cPickle as pickle
from collections import OrderedDict
import numpy as np
import sys, time, os, gzip, theano, math
sys.path.append('../')
from theano import config, printing
theano.config.compute_test_value = 'warn'
from theano.printing import pydotprint
import theano.tensor as T
from utils.misc import saveHDF5
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from evaluate import sampleGaussian
from utils.optimizer import adam, rmsprop
from models.__init__ import BaseModel
from datasets.synthp import params_synthetic
from datasets.synthpTheano import updateParamsSynthetic

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
"""
                                         DEEP MARKOV MODEL [DEEP KALMAN FILTER]
"""


class DKF(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
        self.scan_updates = []
        super(DKF, self).__init__(
            params, paramFile=paramFile, reloadFile=reloadFile)
        if 'synthetic' in self.params['dataset'] and not hasattr(
                self, 'params_synthetic'):
            assert False, 'Expecting to have params_synthetic as an attribute in DKF class'
        assert self.params[
            'nonlinearity'] != 'maxout', 'Maxout nonlinearity not supported'
        if self.params['use_cond']:
            self.params['transition_type'] == 'simple_gated', \
                'Only simple gated transition supports conditioning (dim_cond)'

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _createParams(self):
        """ Model parameters """
        npWeights = OrderedDict()
        self._createInferenceParams(npWeights)
        self._createGenerativeParams(npWeights)
        return npWeights

    def _createGenerativeParams(self, npWeights):
        """ Create weights/params for generative model """
        if 'synthetic' in self.params['dataset']:
            updateParamsSynthetic(params_synthetic)
            self.params_synthetic = params_synthetic
            for k in self.params_synthetic[self.params['dataset']]['params']:
                npWeights[k + '_W'] = np.array(
                    np.random.uniform(-0.2, 0.2), dtype=config.floatX)
            return

        DIM_HIDDEN = self.params['dim_hidden']
        DIM_STOCHASTIC = self.params['dim_stochastic']
        if self.params['use_cond']:
            DIM_COND = self.params['dim_cond']

        # extra dimensions for conditioning transition function
        extra_dims = 0
        if self.params['use_prev_input']:
            extra_dims += self.params['dim_observations']
        if self.params['use_cond']:
            # extra conditioning input at each time step (e.g. a
            # subsequence class)
            extra_dims += DIM_COND

            # for mu and cov at time 0
            npWeights['p_z_init_mu_W'] = self._getWeight(
                (DIM_COND, DIM_STOCHASTIC))
            npWeights['p_z_init_mu_b'] = self._getWeight((DIM_STOCHASTIC, ))
            npWeights['p_z_init_cov_W'] = self._getWeight(
                (DIM_COND, DIM_STOCHASTIC))
            npWeights['p_z_init_cov_b'] = self._getWeight((DIM_STOCHASTIC, ))

        if self.params['transition_type'] == 'mlp':
            DIM_HIDDEN_TRANS = DIM_HIDDEN * 2
            for l in range(self.params['transition_layers']):
                dim_input, dim_output = DIM_HIDDEN_TRANS, DIM_HIDDEN_TRANS
                if l == 0:
                    dim_input = self.params['dim_stochastic']
                npWeights['p_trans_W_' + str(l)] = self._getWeight(
                    (dim_input, dim_output))
                npWeights['p_trans_b_' + str(l)] = self._getWeight(
                    (dim_output, ))
            if extra_dims > 0:
                npWeights['p_trans_W_0'] = self._getWeight(
                    (DIM_STOCHASTIC + extra_dims, DIM_HIDDEN_TRANS))
                npWeights['p_trans_b_0'] = self._getWeight(
                    (DIM_HIDDEN_TRANS, ))
            MU_COV_INP = DIM_HIDDEN_TRANS
        elif self.params['transition_type'] == 'simple_gated':
            DIM_HIDDEN_TRANS = DIM_HIDDEN * 2
            npWeights['p_gate_embed_W_0'] = self._getWeight(
                (DIM_STOCHASTIC, DIM_HIDDEN_TRANS))
            npWeights['p_gate_embed_b_0'] = self._getWeight(
                (DIM_HIDDEN_TRANS, ))
            npWeights['p_gate_embed_W_1'] = self._getWeight(
                (DIM_HIDDEN_TRANS, DIM_STOCHASTIC))
            npWeights['p_gate_embed_b_1'] = self._getWeight((DIM_STOCHASTIC, ))
            npWeights['p_z_W_0'] = self._getWeight(
                (DIM_STOCHASTIC, DIM_HIDDEN_TRANS))
            npWeights['p_z_b_0'] = self._getWeight((DIM_HIDDEN_TRANS, ))
            npWeights['p_z_W_1'] = self._getWeight(
                (DIM_HIDDEN_TRANS, DIM_STOCHASTIC))
            npWeights['p_z_b_1'] = self._getWeight((DIM_STOCHASTIC, ))
            if extra_dims > 0:
                npWeights['p_z_W_0'] = self._getWeight(
                    (DIM_STOCHASTIC + extra_dims, DIM_HIDDEN_TRANS))
                npWeights['p_z_b_0'] = self._getWeight((DIM_HIDDEN_TRANS, ))
                npWeights['p_gate_embed_W_0'] = self._getWeight(
                    (DIM_STOCHASTIC + extra_dims, DIM_HIDDEN_TRANS))
                npWeights['p_gate_embed_b_0'] = self._getWeight(
                    (DIM_HIDDEN_TRANS, ))
            MU_COV_INP = DIM_STOCHASTIC
        else:
            assert False, 'Invalid transition type: ' + self.params[
                'transition_type']

        if self.params['transition_type'] == 'simple_gated':
            weight = np.eye(self.params['dim_stochastic']).astype(
                config.floatX)
            bias = np.zeros(
                (self.params['dim_stochastic'], )).astype(config.floatX)
            # Initialize the weights to be identity
            npWeights['p_trans_W_mu'] = weight
            npWeights['p_trans_b_mu'] = bias
        else:
            npWeights['p_trans_W_mu'] = self._getWeight(
                (MU_COV_INP, self.params['dim_stochastic']))
            npWeights['p_trans_b_mu'] = self._getWeight(
                (self.params['dim_stochastic'], ))
        npWeights['p_trans_W_cov'] = self._getWeight(
            (MU_COV_INP, self.params['dim_stochastic']))
        npWeights['p_trans_b_cov'] = self._getWeight(
            (self.params['dim_stochastic'], ))

        # Emission Function [MLP]
        if self.params['emission_type'] == 'mlp':
            for l in range(self.params['emission_layers']):
                dim_input, dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l == 0:
                    dim_input = self.params['dim_stochastic']
                npWeights['p_emis_W_' + str(l)] = self._getWeight(
                    (dim_input, dim_output))
                npWeights['p_emis_b_' + str(l)] = self._getWeight(
                    (dim_output, ))
        elif self.params['emission_type'] == 'res':
            for l in range(self.params['emission_layers']):
                dim_input, dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l == 0:
                    dim_input = self.params['dim_stochastic']
                npWeights['p_emis_W_' + str(l)] = self._getWeight(
                    (dim_input, dim_output))
                npWeights['p_emis_b_' + str(l)] = self._getWeight(
                    (dim_output, ))
            dim_res_out = self.params['dim_observations']
            if self.params['data_type'] == 'binary_nade':
                dim_res_out = DIM_HIDDEN
            npWeights['p_res_W'] = self._getWeight(
                (self.params['dim_stochastic'], dim_res_out))
        elif self.params['emission_type'] == 'conditional':
            for l in range(self.params['emission_layers']):
                dim_input, dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l == 0:
                    dim_input = self.params['dim_stochastic'] + self.params[
                        'dim_observations']
                npWeights['p_emis_W_' + str(l)] = self._getWeight(
                    (dim_input, dim_output))
                npWeights['p_emis_b_' + str(l)] = self._getWeight(
                    (dim_output, ))
        else:
            assert False, 'Invalid emission type: ' + str(
                self.params['emission_type'])

        if self.params['data_type'] == 'binary':
            npWeights['p_emis_W_ber'] = self._getWeight(
                (self.params['dim_hidden'], self.params['dim_observations']))
            npWeights['p_emis_b_ber'] = self._getWeight(
                (self.params['dim_observations'], ))
        elif self.params['data_type'] == 'real':
            npWeights['p_emis_W_mu'] = self._getWeight(
                (self.params['dim_hidden'], self.params['dim_observations']))
            npWeights['p_emis_b_mu'] = self._getWeight(
                (self.params['dim_observations'], ))
            npWeights['p_emis_W_var'] = self._getWeight(
                (self.params['dim_hidden'], self.params['dim_observations']))
            npWeights['p_emis_b_var'] = self._getWeight(
                (self.params['dim_observations'], ))
        elif self.params['data_type'] == 'binary_nade':
            n_visible, n_hidden = self.params['dim_observations'], self.params[
                'dim_hidden']
            npWeights['p_nade_W'] = self._getWeight((n_visible, n_hidden))
            npWeights['p_nade_U'] = self._getWeight((n_visible, n_hidden))
            npWeights['p_nade_b'] = self._getWeight((n_visible, ))
        else:
            assert False, 'Invalid datatype: ' + self.params['data_type']

    def _createInferenceParams(self, npWeights):
        """  Create weights/params for inference network """

        # Initial embedding for the inputs
        DIM_INPUT = self.params['dim_observations']
        if self.params['use_cond']:
            DIM_INPUT += self.params['dim_cond']
        RNN_SIZE = self.params['rnn_size']

        DIM_HIDDEN = RNN_SIZE
        DIM_STOC = self.params['dim_stochastic']

        # Embed the Input -> RNN_SIZE
        dim_input, dim_output = DIM_INPUT, RNN_SIZE
        npWeights['q_W_input_0'] = self._getWeight((dim_input, dim_output))
        npWeights['q_b_input_0'] = self._getWeight((dim_output, ))

        # Setup weights for LSTM
        self._createLSTMWeights(npWeights)

        # Embedding before MF/ST inference model
        if self.params['inference_model'] == 'mean_field':
            pass
        elif self.params['inference_model'] == 'structured':
            DIM_INPUT = self.params['dim_stochastic']
            if self.params['use_generative_prior']:
                DIM_INPUT = self.params['rnn_size']
            npWeights['q_W_st_0'] = self._getWeight(
                (DIM_INPUT, self.params['rnn_size']))
            npWeights['q_b_st_0'] = self._getWeight(
                (self.params['rnn_size'], ))
        else:
            assert False, 'Invalid inference model: ' + self.params[
                'inference_model']
        RNN_SIZE = self.params['rnn_size']
        npWeights['q_W_mu'] = self._getWeight(
            (RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_mu'] = self._getWeight(
            (self.params['dim_stochastic'], ))
        npWeights['q_W_cov'] = self._getWeight(
            (RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_cov'] = self._getWeight(
            (self.params['dim_stochastic'], ))
        if self.params['var_model'] == 'LR' and self.params[
                'inference_model'] == 'mean_field':
            npWeights['q_W_mu_r'] = self._getWeight(
                (RNN_SIZE, self.params['dim_stochastic']))
            npWeights['q_b_mu_r'] = self._getWeight(
                (self.params['dim_stochastic'], ))
            npWeights['q_W_cov_r'] = self._getWeight(
                (RNN_SIZE, self.params['dim_stochastic']))
            npWeights['q_b_cov_r'] = self._getWeight(
                (self.params['dim_stochastic'], ))

    def _createLSTMWeights(self, npWeights):
        # LSTM L/LR/R w/ orthogonal weight initialization
        suffices_to_build = []
        if self.params['var_model'] == 'LR' or self.params['var_model'] == 'L':
            suffices_to_build.append('l')
        if self.params['var_model'] == 'LR' or self.params['var_model'] == 'R':
            suffices_to_build.append('r')
        RNN_SIZE = self.params['rnn_size']
        for suffix in suffices_to_build:
            for l in range(self.params['rnn_layers']):
                npWeights['W_lstm_' + suffix + '_' + str(l)] = self._getWeight(
                    (RNN_SIZE, RNN_SIZE * 4))
                npWeights['b_lstm_' + suffix + '_' + str(l)] = self._getWeight(
                    (RNN_SIZE * 4, ), scheme='lstm')
                npWeights['U_lstm_' + suffix + '_' + str(l)] = self._getWeight(
                    (RNN_SIZE, RNN_SIZE * 4), scheme='lstm')

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _getEmissionFxn(self, z, X=None):
        """
        Apply emission function to zs
        Input:  z [bs x T x dim]
        Output: (params, ) or (mu, cov) of size [bs x T x dim]
        """
        if 'synthetic' in self.params['dataset']:
            self._p('Using emission function for ' + self.params['dataset'])
            tParams = {}
            for k in self.params_synthetic[self.params['dataset']]['params']:
                tParams[k] = self.tWeights[k + '_W']
            mu = self.params_synthetic[self.params['dataset']]['obs_fxn'](
                z, fxn_params=tParams)
            cov = T.ones_like(mu) * self.params_synthetic[self.params[
                'dataset']]['obs_cov']
            cov.name = 'EmissionCov'
            return [mu, cov]

        if self.params['emission_type'] in ['mlp', 'res']:
            self._p('EMISSION TYPE: MLP or RES')
            hid = z
        elif self.params['emission_type'] == 'conditional':
            self._p('EMISSION TYPE: conditional')
            # TODO: if this doesn't work then you should make it a gated unit
            # (like a GRU)
            X_prev = T.concatenate(
                [T.zeros_like(X[:, [0], :]), X[:, :-1, :]], axis=1)
            hid = T.concatenate([z, X_prev], axis=2)
        else:
            assert False, 'Invalid emission type'
        for l in range(self.params['emission_layers']):
            if self.params['data_type'] == 'binary_nade' and l == self.params[
                    'emission_layers'] - 1:
                hid = T.dot(
                    hid, self.tWeights['p_emis_W_' +
                                       str(l)]) + self.tWeights['p_emis_b_' +
                                                                str(l)]
            else:
                hid = self._LinearNL(self.tWeights['p_emis_W_' + str(l)],
                                     self.tWeights['p_emis_b_' + str(l)], hid)

        if self.params['data_type'] == 'binary':
            if self.params['emission_type'] == 'res':
                hid = T.dot(z, self.tWeights['p_res_W']) + T.dot(
                    hid, self.tWeights['p_emis_W_ber']) + self.tWeights[
                        'p_emis_b_ber']
                mean_params = T.nnet.sigmoid(hid)
            else:
                mean_params = T.nnet.sigmoid(
                    T.dot(hid, self.tWeights['p_emis_W_ber']) + self.tWeights[
                        'p_emis_b_ber'])
            return [mean_params]
        elif self.params['data_type'] == 'real':
            # for real vars, we use linear mean activation and log(1+exp(x))
            # variance activation
            if self.params['emission_type'] == 'res':
                # I've left the 'residual' (? is that it) thing out of the
                # variance calculation; I don't think it makes sense there
                # (even though it makes perfect sense for the mean)---at least,
                # that squares with the SRNN paper (Sonderby et al.?)
                res_out = T.dot(z, self.tWeights['p_res_W'])
                hid_mu = res_out + T.dot(hid, self.tWeights['p_emis_W_mu']) \
                         + self.tWeights['p_emis_b_mu']
                hid_var = T.nnet.softplus(
                    T.dot(hid, self.tWeights['p_emis_W_var']) + self.tWeights[
                        'p_emis_b_var'])
            else:
                hid_mu = T.dot(hid, self.tWeights[
                    'p_emis_W_mu']) + self.tWeights['p_emis_b_mu']
                hid_var = T.nnet.softplus(
                    T.dot(hid, self.tWeights['p_emis_W_var']) + self.tWeights[
                        'p_emis_b_var'])
            return (hid_mu, hid_var)
        elif self.params['data_type'] == 'binary_nade':
            self._p('NADE observations')
            assert X is not None, 'Need observations for NADE'
            if self.params['emission_type'] == 'res':
                hid += T.dot(z, self.tWeights['p_res_W'])
            x_reshaped = X.dimshuffle(2, 0, 1)
            x0 = T.ones((hid.shape[0], hid.shape[1]))  #x_reshaped[0]) # bs x T
            a0 = hid  #bs x T x nhid
            W = self.tWeights['p_nade_W']
            V = self.tWeights['p_nade_U']
            b = self.tWeights['p_nade_b']

            # Use a NADE at the output
            def NADEDensity(x, w, v, b, a_prev,
                            x_prev):  #Estimating likelihood
                a = a_prev + T.dot(
                    T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
                h = T.nnet.sigmoid(a)  #Original - bs x T x nhid
                p_xi_is_one = T.nnet.sigmoid(T.dot(h, v) + b)
                return (a, x, p_xi_is_one)

            ([_, _, mean_params], _) = theano.scan(
                NADEDensity,
                sequences=[x_reshaped, W, V, b],
                outputs_info=[a0, x0, None])

            # theano function to sample from NADE
            def NADESample(w, v, b, a_prev_s, x_prev_s):
                a_s = a_prev_s + T.dot(
                    T.shape_padright(x_prev_s, 1), T.shape_padleft(w, 1))
                h_s = T.nnet.sigmoid(a_s)  #Original - bs x T x nhid
                p_xi_is_one_s = T.nnet.sigmoid(T.dot(h_s, v) + b)
                x_s = T.switch(p_xi_is_one_s > 0.5, 1., 0.)
                return (a_s, x_s, p_xi_is_one_s)

            ([_, _, sampled_params], _) = theano.scan(
                NADESample, sequences=[W, V, b], outputs_info=[a0, x0, None])
            """
            def NADEDensityAndSample(x, w, v, b,
                                     a_prev,   x_prev,
                                     a_prev_s, x_prev_s ):
                a     = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
                h     = T.nnet.sigmoid(a) #bs x T x nhid
                p_xi_is_one = T.nnet.sigmoid(T.dot(h, v) + b)

                a_s   = a_prev_s + T.dot(T.shape_padright(x_prev_s, 1), T.shape_padleft(w, 1))
                h_s   = T.nnet.sigmoid(a_s) #bs x T x nhid
                p_xi_is_one_s = T.nnet.sigmoid(T.dot(h_s, v) + b)
                x_s   = T.switch(p_xi_is_one_s>0.5,1.,0.)
                return (a, x, a_s, x_s, p_xi_is_one, p_xi_is_one_s)

            ([_, _, _, _, mean_params,sampled_params], _) = theano.scan(NADEDensityAndSample,
                                                   sequences=[x_reshaped, W, V, b],
                                                   outputs_info=[a0, x0, a0, x0, None, None])
            """
            sampled_params = sampled_params.dimshuffle(1, 2, 0)
            mean_params = mean_params.dimshuffle(1, 2, 0)
            return [mean_params, sampled_params]
        else:
            assert False, 'Invalid type of data'

    def _getTransitionFxn(self, z, X=None, U=None):
        """
        Apply transition function to zs
        Input:  z [bs x T x dim],
                X<if conditioning on observations>[bs x T x dim],
                U<if actions present in model> [bs x T x dim]
        Output: mu, cov of size [bs x T x dim]
        """
        if 'synthetic' in self.params['dataset']:
            self._p('Using transition function for ' + self.params['dataset'])
            tParams = {}
            for k in self.params_synthetic[self.params['dataset']]['params']:
                tParams[k] = self.tWeights[k + '_W']
            mu = self.params_synthetic[self.params['dataset']]['trans_fxn'](
                z, fxn_params=tParams)
            cov = T.ones_like(mu) * self.params_synthetic[self.params[
                'dataset']]['trans_cov']
            cov.name = 'TransitionCov'
            return mu, cov

        if self.params['transition_type'] == 'simple_gated':

            def mlp(inp, W1, b1, W2, b2, X_prev=None):
                if X_prev is not None:
                    cat_res = T.concatenate([inp, X_prev], axis=2)
                    h1 = self._LinearNL(W1, b1, cat_res)
                else:
                    h1 = self._LinearNL(W1, b1, inp)
                # XXX: the activations are exploding here sometimes. Likely due
                # to activation inside the MLP for h1. Making it a sigmoid
                # instead of a ReLU will likely fix the problem (requires
                # _linearNL changes).
                h2 = T.dot(h1, W2) + b2
                return h2

            # we can pass in previous observation and/or conditioning information (U)
            X_prev_list = []
            if self.params['use_prev_input']:
                prev_input = T.concatenate(
                    [T.zeros_like(X[:, [0], :]), X[:, :-1, :]], axis=1)
                X_prev_list.append(prev_input)
            if self.params['use_cond']:
                if self.params.get('gumbel_softmax_cond', False):
                    U_sample = self._gumbel_softmax(U)
                else:
                    U_sample = U
                X_prev_list.append(U_sample)
            if X_prev_list:
                X_prev = T.concatenate(X_prev_list, axis=2)
            else:
                X_prev = None

            gateInp = z
            gate = T.nnet.sigmoid(
                mlp(gateInp,
                    self.tWeights['p_gate_embed_W_0'],
                    self.tWeights['p_gate_embed_b_0'],
                    self.tWeights['p_gate_embed_W_1'],
                    self.tWeights['p_gate_embed_b_1'],
                    X_prev=X_prev))
            z_prop = mlp(z,
                         self.tWeights['p_z_W_0'],
                         self.tWeights['p_z_b_0'],
                         self.tWeights['p_z_W_1'],
                         self.tWeights['p_z_b_1'],
                         X_prev=X_prev)
            mu = gate * z_prop + (1. - gate) * (T.dot(
                z, self.tWeights['p_trans_W_mu']) +
                                                self.tWeights['p_trans_b_mu'])
            cov = T.nnet.softplus(
                T.dot(self._applyNL(z_prop), self.tWeights['p_trans_W_cov']) +
                self.tWeights['p_trans_b_cov'])
            return mu, cov
        elif self.params['transition_type'] == 'mlp':
            hid = z
            if self.params['use_prev_input']:
                X_prev = T.concatenate(
                    [T.zeros_like(X[:, [0], :]), X[:, :-1, :]], axis=1)
                hid = T.concatenate([hid, X_prev], axis=2)
            for l in range(self.params['transition_layers']):
                hid = self._LinearNL(self.tWeights['p_trans_W_' + str(l)],
                                     self.tWeights['p_trans_b_' + str(l)], hid)
            mu = T.dot(
                hid,
                self.tWeights['p_trans_W_mu']) + self.tWeights['p_trans_b_mu']
            cov = T.nnet.softplus(
                T.dot(hid, self.tWeights['p_trans_W_cov']) + self.tWeights[
                    'p_trans_b_cov'])
            return mu, cov
        else:
            assert False, 'Invalid Transition type: ' + str(
                self.params['transition_type'])

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _buildLSTM(self, X, embedding, dropout_prob=0.):
        """
        Take the embedding of bs x T x dim and return T x bs x dim that is the result of the scan operation
        for L/LR/R
        Input: embedding [bs x T x dim]
        Output:hidden_state [T x bs x dim]
        """
        start_time = time.time()
        self._p('In <_buildLSTM>')
        suffix = ''
        if self.params['var_model'] == 'R':
            suffix = 'r'
            return self._LSTMlayer(embedding, suffix, dropout_prob)
        elif self.params['var_model'] == 'L':
            suffix = 'l'
            return self._LSTMlayer(embedding, suffix, dropout_prob)
        elif self.params['var_model'] == 'LR':
            suffix = 'l'
            l2r = self._LSTMlayer(embedding, suffix, dropout_prob)
            suffix = 'r'
            r2l = self._LSTMlayer(embedding, suffix, dropout_prob)
            return [l2r, r2l]
        else:
            assert False, 'Invalid variational model'
        self._p(('Done <_buildLSTM> [Took %.4f]') % (time.time() - start_time))

    def _inferenceLayer(self, hidden_state):
        """
        Take input of T x bs x dim and return z, mu,
        sq each of size (bs x T x dim)
        Input: hidden_state [T x bs x dim], eps [bs x T x dim]
        Output: z [bs x T x dim], mu [bs x T x dim], cov [bs x T x dim]
        """

        def structuredApproximation(h_t, eps_t, z_prev, q_W_st_0, q_b_st_0,
                                    q_W_mu, q_b_mu, q_W_cov, q_b_cov):
            # Using the prior distribution directly
            if self.params['use_generative_prior']:
                assert not self.params[
                    'use_prev_input'], 'No support for using previous input'
                assert 'use_cond' not in self.params, "No support for extra conditioning"
                # Get mu/cov from z_prev through prior distribution
                mu_1, cov_1 = self._getTransitionFxn(z_prev)
                # Combine with estimate of mu/cov from data
                h_data = T.tanh(T.dot(h_t, q_W_st_0) + q_b_st_0)
                mu_2 = T.dot(h_data, q_W_mu) + q_b_mu
                cov_2 = T.nnet.softplus(T.dot(h_data, q_W_cov) + q_b_cov)
                mu = (mu_1 * cov_2 + mu_2 * cov_1) / (cov_1 + cov_2)
                cov = (cov_1 * cov_2) / (cov_1 + cov_2)
                z = mu + T.sqrt(cov) * eps_t
                return z, mu, cov
            else:
                h_next = T.tanh(T.dot(z_prev, q_W_st_0) + q_b_st_0)
                if self.params['var_model'] == 'LR':
                    h_next = (1. / 3.) * (h_t + h_next)
                else:
                    h_next = (1. / 2.) * (h_t + h_next)
                mu_t = T.dot(h_next, q_W_mu) + q_b_mu
                cov_t = T.nnet.softplus(T.dot(h_next, q_W_cov) + q_b_cov)
                z_t = mu_t + T.sqrt(cov_t) * eps_t
                return z_t, mu_t, cov_t

        if type(hidden_state) is list:
            eps = self.srng.normal(size=(hidden_state[0].shape[1],
                                         hidden_state[0].shape[0],
                                         self.params['dim_stochastic']))
        else:
            eps = self.srng.normal(size=(hidden_state.shape[1],
                                         hidden_state.shape[0],
                                         self.params['dim_stochastic']))
        if self.params['inference_model'] == 'structured':
            # Structured recognition networks
            if self.params['var_model'] == 'LR':
                state = hidden_state[0] + hidden_state[1]
            else:
                state = hidden_state
            eps_swap = eps.swapaxes(0, 1)
            if self.params['dim_stochastic'] == 1:
                """
                TODO: Write to theano authors regarding this issue.
                Workaround for theano issue: The result of a matrix multiply is
                a "matrix" even if one of the dimensions is 1. However defining
                a tensor with one dimension one means theano regards the
                resulting tensor as a matrix and consequently in the scan as a
                column. This results in a mismatch in tensor type in input
                (column) and output (matrix) and throws an error. This is a
                workaround that preserves type while not affecting dimensions
                """
                z0 = T.zeros((eps_swap.shape[1], self.params['rnn_size']))
                z0 = T.dot(z0, T.zeros_like(self.tWeights['q_W_mu']))
            else:
                z0 = T.zeros(
                    (eps_swap.shape[1], self.params['dim_stochastic']))
            rval, _ = theano.scan(
                structuredApproximation,
                sequences=[state, eps_swap],
                outputs_info=[z0, None, None],
                non_sequences=[
                    self.tWeights[k] for k in ['q_W_st_0', 'q_b_st_0']
                ] + [
                    self.tWeights[k]
                    for k in ['q_W_mu', 'q_b_mu', 'q_W_cov', 'q_b_cov']
                ],
                name='structuredApproximation')
            z, mu, cov = rval[0].swapaxes(0, 1), rval[1].swapaxes(
                0, 1), rval[2].swapaxes(0, 1)
            return z, mu, cov
        elif self.params['inference_model'] == 'mean_field':
            if self.params['var_model'] == 'LR':
                l2r = hidden_state[0].swapaxes(0, 1)
                r2l = hidden_state[1].swapaxes(0, 1)
                hidl2r = l2r
                mu_1 = T.dot(hidl2r,
                             self.tWeights['q_W_mu']) + self.tWeights['q_b_mu']
                cov_1 = T.nnet.softplus(
                    T.dot(hidl2r, self.tWeights['q_W_cov']) + self.tWeights[
                        'q_b_cov'])
                hidr2l = r2l
                mu_2 = T.dot(
                    hidr2l,
                    self.tWeights['q_W_mu_r']) + self.tWeights['q_b_mu_r']
                cov_2 = T.nnet.softplus(
                    T.dot(hidr2l, self.tWeights['q_W_cov_r']) + self.tWeights[
                        'q_b_cov_r'])
                mu = (mu_1 * cov_2 + mu_2 * cov_1) / (cov_1 + cov_2)
                cov = (cov_1 * cov_2) / (cov_1 + cov_2)
                z = mu + T.sqrt(cov) * eps
            else:
                hid = hidden_state.swapaxes(0, 1)
                mu = T.dot(hid,
                           self.tWeights['q_W_mu']) + self.tWeights['q_b_mu']
                cov = T.nnet.softplus(
                    T.dot(hid, self.tWeights['q_W_cov']) + self.tWeights[
                        'q_b_cov'])
                z = mu + T.sqrt(cov) * eps
            return z, mu, cov
        else:
            assert False, 'Invalid recognition model'

    def _gumbel_softmax(self, U):
        """Return Gumbel-Softmax samples of some categorical distribution."""
        assert False, "Are you sure you meant to use this? Applying it to " \
            "an input layer (its original purpose) is not useful because " \
            "you don't need to backprop through input layers. Right now " \
            "2017-03-27) I don't know of anywhere else it can be used."
        # turn uni(0, 1) into gumbel(0, 1)
        uni_rand = self.srng.uniform(low=0., high=1., size=U.shape)
        gumb_rand = -T.log(-T.log(uni_rand + 1e-10) + 1e-20)
        temp = self.tWeights['cond_temp']
        # pad U for stability
        pre_soft = (gumb_rand + T.log(U + 1e-20)) / temp

        # theano can only softmax matrices, so we need a reshape
        new_shape = (T.prod(U.shape[:-1]), U.shape[-1])
        flat = pre_soft.reshape(new_shape)
        maxed = T.nnet.softmax(flat)
        rv = maxed.reshape(U.shape)

        return rv

    def _qEmbeddingLayer(self, X, U=None):
        """ Embed for q """
        if self.params['use_cond']:
            if self.params.get('gumbel_softmax_cond', False):
                U_samples = self._gumbel_softmax(U)
            else:
                U_samples = U
            P = T.concatenate((X, U_samples), axis=2)
        else:
            P = X
        return self._LinearNL(self.tWeights['q_W_input_0'],
                              self.tWeights['q_b_input_0'], P)

    def _getInitPriorFxn(self, X, U=None):
        # gives mu and cov at t = 0
        # v XXX: Disabled because I want to keep using class before upgrade.
        # Fix this!
        # if not self.params['use_cond']:
        if True:
            # v XXX: also remove this once done, I guess
            # assert U is None, "shouldn't be getting a U w/ non-cond model"

            # constant because we have no other information to use
            mu_prior0 = T.alloc(
                np.asarray(0.).astype(config.floatX), X.shape[0], 1,
                self.params['dim_stochastic'])
            cov_prior0 = T.alloc(
                np.asarray(1.).astype(config.floatX), X.shape[0], 1,
                self.params['dim_stochastic'])
        else:
            assert U is not None, "need U with use_cond == True"

            # U-conditioned with single-layer network; if U is one-hot then
            # this allows us to learn different mu/cov for each value.
            p_mu_W = self.tWeights['p_z_init_mu_W']
            p_mu_b = self.tWeights['p_z_init_mu_b']
            p_cov_W = self.tWeights['p_z_init_cov_W']
            p_cov_b = self.tWeights['p_z_init_cov_b']
            mu_prior0 = T.dot(U, p_mu_W) + p_mu_b
            cov_prior0 = T.nnet.softplus(T.dot(U, p_cov_W) + p_cov_b)

        return mu_prior0, cov_prior0

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

    def _inferenceAndReconstruction(self, X, dropout_prob=0., U=None):
        """
        Returns z_q, mu_q and cov_q
        """
        self._p('Building with dropout:' + str(dropout_prob))
        embedding = self._qEmbeddingLayer(X, U)
        hidden_state = self._buildLSTM(X, embedding, dropout_prob)
        z_q, mu_q, cov_q = self._inferenceLayer(hidden_state)

        # Regularize z_q (for train)
        # if dropout_prob>0.:
        #    z_q  = z_q + self.srng.normal(z_q.shape, 0.,0.0025,dtype=config.floatX)
        # z_q.name          = 'z_q'

        observation_params = self._getEmissionFxn(z_q, X=X)
        mu_trans, cov_trans = self._getTransitionFxn(z_q, X=X, U=U)
        # TODO: replace zeroth prior mean/covariance with something that's
        # actually learnt. Make sure you update other pieces of code which rely
        # on these quantities.
        mu_prior0, cov_prior0 = self._getInitPriorFxn(X, U)
        mu_prior = T.concatenate([mu_prior0, mu_trans[:, :-1, :]], axis=1)
        cov_prior = T.concatenate([cov_prior0, cov_trans[:, :-1, :]], axis=1)

        # Guide to return values: mu_q, cov_q define q_\phi(z_t \mid z_{t-1},
        # x) in the paper. mu_prior, cov_prior define p_\theta(z_t \mid
        # z_{t-1}). One of the objectives is to minimise the KL between those
        # two---that trains both the prior transition model p and the posterior
        # (i.e. observation-conditioned) transition model q.

        return observation_params, z_q, mu_q, cov_q, mu_prior, cov_prior, mu_trans, cov_trans

    def _getTemporalKL(self,
                       mu_q,
                       cov_q,
                       mu_prior,
                       cov_prior,
                       M2D,
                       batchVector=False):
        """
        TemporalKL divergence KL (q||p)
        KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q)
                        + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
        M2D is a mask of size bs x T that should be applied once the KL divergence for each point
        across time has been estimated
        """
        assert np.all(cov_q.tag.test_value > 0.), 'should be positive'
        assert np.all(cov_prior.tag.test_value > 0.), 'should be positive'
        diff_mu = mu_prior - mu_q
        KL_t = T.log(cov_prior) - T.log(
            cov_q) - 1. + cov_q / cov_prior + diff_mu**2 / cov_prior
        KLvec = (0.5 * KL_t.sum(2) * M2D).sum(1, keepdims=True)
        if batchVector:
            return KLvec
        else:
            return KLvec.sum()

    def _getNegCLL(self, obs_params, X, M, batchVector=False):
        """
        Estimate the negative conditional log likelihood of x|z under the generative model
        M: mask of size bs x T or bs x T x D
        X: target of size bs x T x dim
        """
        m_dim = M.ndim
        if self.params['data_type'] == 'real':
            mu_p = obs_params[0]
            cov_p = obs_params[1]
            std_p = T.sqrt(cov_p)
            negCLL_t = 0.5 * np.log(2 * np.pi) + 0.5 * T.log(cov_p) + 0.5 * (
                (X - mu_p) / std_p)**2
            if m_dim == 3:
                negCLL = (negCLL_t * M).sum(1, keepdims=True)
            else:
                negCLL = (negCLL_t.sum(2) * M).sum(1, keepdims=True)
        else:
            mean_p = obs_params[0]
            if m_dim == 3:
                negCLL = (T.nnet.binary_crossentropy(mean_p, X) * M).sum(
                    1, keepdims=True)
            else:
                negCLL = (T.nnet.binary_crossentropy(mean_p, X).sum(2) *
                          M).sum(
                              1, keepdims=True)
        if batchVector:
            return negCLL
        else:
            return negCLL.sum()

    def _getL1Aux(self, l1_lam, obs_params, X, M, batchVector=False):
        """Compute an L1-based auxiliary loss w/ coefficient lam."""
        m_dim = M.ndim
        if self.params['data_type'] == 'real':
            mu_p = obs_params[0]
            diff = abs(mu_p - X)
            if m_dim == 3:
                aux_loss = l1_lam * (diff * M).sum(1, keepdims=True)
            else:
                aux_loss = l1_lam * (diff.sum(2) * M).sum(
                    1, keepdims=True)
        else:
            raise ValueError('only does L1 auxiliary for real outs right now')
        if batchVector:
            return aux_loss
        else:
            return aux_loss.sum()

    def resetDataset(self, newX, newM, newU=None, quiet=False):
        if not quiet:
            dims = self.dimData()
            self._p('Original dim:' + ', '.join(map(str, dims)))
        assert newM.ndim in (2, 3), newM.ndim
        if newM.ndim == 2:
            if not quiet:
                self._p('Broadcasting new mask up to three dimensions')
            newM = np.tile(newM[..., np.newaxis],
                           [1, 1, self.params['dim_observations']])
        kwargs = dict(
            newX=newX.astype(config.floatX),
            newMask=newM.astype(config.floatX))
        assert (newU is not None) == bool(self.params['use_cond']), \
            'need U iff use_cond given'
        if self.params['use_cond']:
            kwargs['newU'] = newU
        self.setData(**kwargs)
        if not quiet:
            dims = self.dimData()
            self._p('New dim:' + ', '.join(map(str, dims)))

    def _buildModel(self):
        if 'synthetic' in self.params['dataset']:
            self.params_synthetic = params_synthetic
        """ High level function to build and setup theano functions """
        # X      = T.tensor3('X',   dtype=config.floatX)
        # eps    = T.tensor3('eps', dtype=config.floatX)
        # M      = T.tensor3('M', dtype=config.floatX)
        idx = T.vector('idx', dtype='int64')
        idx.tag.test_value = np.array([0, 1]).astype('int64')
        self.dataset = theano.shared(
            np.random.uniform(
                0, 1, size=(3, 5, self.params['dim_observations'])).astype(
                    config.floatX))
        # same as dataset, but [0, 1]-constrained
        self.mask = theano.shared((0.5 <= np.random.uniform(
            0, 1, size=(3, 5, self.params['dim_observations']))
                                   ).astype(config.floatX))
        X_o = self.dataset[idx]
        M_o = self.mask[idx]
        # should be 1D
        maxidx = T.cast(M_o.any(axis=2).sum(axis=1).max(axis=0), 'int64')
        X = X_o[:, :maxidx, :]
        M = M_o[:, :maxidx]
        M2D = M.all(axis=2).astype(config.floatX)
        newX, newMask = T.tensor3(
            'newX', dtype=config.floatX), T.tensor3(
                'newMask', dtype=config.floatX)
        set_data_inputs = [newX, newMask]
        set_data_updates = [(self.dataset, newX), (self.mask, newMask)]
        set_data_shapes = [self.dataset.shape, self.mask.shape]
        if self.params['use_cond']:
            self.cond_vals = theano.shared(
                np.random.uniform(0, 1, size=(3, 5, self.params['dim_cond']))
                .astype(config.floatX))
            U_o = self.cond_vals[idx]
            U = U_o[:, :maxidx]
            newU = T.tensor3('newU', dtype=config.floatX)
            set_data_inputs.append(newU)
            set_data_updates.append((self.cond_vals, newU))
            set_data_shapes.append(self.cond_vals.shape)

            # temperature for Gumbel-Softmax; initial value taken from concrete
            # distribution paper
            # TODO: Figure out the best way of supplying/annealing this.
            start_temp = 1.0 / (self.params['dim_cond'] - 1)
            self._addWeights(
                'cond_temp',
                np.asarray(start_temp, dtype=config.floatX),
                borrow=False)
        else:
            U = None
        self.setData = theano.function(
            set_data_inputs, None, updates=set_data_updates)
        self.dimData = theano.function([], set_data_shapes)

        # Learning Rates and annealing objective function
        # Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights(
            'lr',
            np.asarray(self.params['lr'], dtype=config.floatX),
            borrow=False)
        self._addWeights(
            'anneal', np.asarray(0.01, dtype=config.floatX), borrow=False)
        self._addWeights(
            'update_ctr', np.asarray(1., dtype=config.floatX), borrow=False)
        use_l1 = bool(self.params.get('l1'))
        if use_l1:
            self._addWeights(
                'l1',
                np.asarray(self.params['l1'], dtype=config.floatX),
                borrow=False)
            l1_lam = self.tWeights['l1']
        lr = self.tWeights['lr']
        anneal = self.tWeights['anneal']
        iteration_t = self.tWeights['update_ctr']

        anneal_div = 1000.
        if 'anneal_rate' in self.params:
            self._p('Anneal = 1 in ' + str(self.params['anneal_rate']) +
                    ' param. updates')
            anneal_div = self.params['anneal_rate']
        if 'synthetic' in self.params['dataset']:
            anneal_div = 100.
        anneal_update = [(iteration_t, iteration_t + 1), (
            anneal, T.switch(0.01 + iteration_t / anneal_div > 1, 1,
                             0.01 + iteration_t / anneal_div))]
        fxn_inputs = [idx]
        if not self.params['validate_only']:
            print('****** CREATING TRAINING FUNCTION*****')
            # ############ Setup training functions ###########
            obs_params, z_q, mu_q, cov_q, mu_prior, cov_prior, _, _ \
                = self._inferenceAndReconstruction(
                    X, dropout_prob=self.params['rnn_dropout'], U=U)
            negCLL = self._getNegCLL(obs_params, X, M)
            # XXX: does this consider the extra KL which comes from z_0? If not,
            # should make it consider that. If so, should update so that it
            # measures KL against action-conditioned prior for z_0.
            TemporalKL = self._getTemporalKL(mu_q, cov_q, mu_prior, cov_prior,
                                             M2D)
            train_cost = negCLL + anneal * TemporalKL
            if use_l1:
                self._p('using l1 reg with lam = %f' % self.params['l1'])
                l1_term = self._getL1Aux(l1_lam, obs_params, X, M)
                train_cost += l1_term
            else:
                l1_term = T.as_tensor_variable(0)

            # Get updates from optimizer
            model_params = self._getModelParams()
            optimizer_up, norm_list = self._setupOptimizer(
                train_cost,
                model_params,
                lr=lr,
                # Turning off for synthetic
                # reg_type =self.params['reg_type'],
                # reg_spec =self.params['reg_spec'],
                # reg_value= self.params['reg_value'],
                divide_grad=T.cast(X.shape[0], dtype=config.floatX),
                grad_norm=1.)

            # Add annealing updates
            optimizer_up += anneal_update + self.updates
            self._p(str(len(self.updates)) + ' other updates')
            # ############ Setup train & evaluate functions ###########
            self.train_debug = theano.function(
                fxn_inputs, [
                    train_cost, norm_list[0], norm_list[1], norm_list[2],
                    negCLL, TemporalKL, anneal.sum(), l1_term
                ],
                updates=optimizer_up,
                name='Train (with Debug)')
        # Updates ack
        self.updates_ack = True
        eval_obs_params, eval_z_q, eval_mu_q, eval_cov_q, eval_mu_prior, eval_cov_prior, \
            eval_mu_trans, eval_cov_trans = self._inferenceAndReconstruction(X,dropout_prob = 0., U=U)
        eval_z_q.name = 'eval_z_q'
        eval_CNLLvec = self._getNegCLL(eval_obs_params, X, M, batchVector=True)
        eval_KLvec = self._getTemporalKL(
            eval_mu_q,
            eval_cov_q,
            eval_mu_prior,
            eval_cov_prior,
            M2D,
            batchVector=True)
        eval_cost = eval_CNLLvec + eval_KLvec
        if use_l1:
            eval_l1_term = self._getL1Aux(l1_lam, eval_obs_params, X, M, batchVector=True)
            eval_cost += eval_l1_term

        # From here on, convert to the log covariance since we only use it for evaluation
        assert np.all(eval_cov_q.tag.test_value > 0.), 'should be positive'
        assert np.all(eval_cov_prior.tag.test_value > 0.), 'should be positive'
        assert np.all(eval_cov_trans.tag.test_value > 0.), 'should be positive'
        eval_logcov_q = T.log(eval_cov_q)
        eval_logcov_prior = T.log(eval_cov_prior)
        eval_logcov_trans = T.log(eval_cov_trans)

        ll_prior = (self._llGaussian(eval_z_q, eval_mu_prior,
                                     eval_logcov_prior)).sum(2) * M2D
        ll_posterior = (self._llGaussian(eval_z_q, eval_mu_q,
                                         eval_logcov_q)).sum(2) * M2D
        ll_estimate = -1 * eval_CNLLvec + ll_prior.sum(
            1, keepdims=True) - ll_posterior.sum(
                1, keepdims=True)

        eval_inputs = [eval_z_q]
        if self.params['use_cond']:
            eval_inputs.append(U)
        self.likelihood = theano.function(
            fxn_inputs,
            ll_estimate,
            name='Importance Sampling based likelihood')
        self.evaluate = theano.function(
            fxn_inputs, eval_cost, name='Evaluate Bound')
        if self.params['use_prev_input']:
            eval_inputs.append(X)
        self.transition_fxn = theano.function(
            eval_inputs, [eval_mu_trans, eval_logcov_trans],
            name='Transition Function')
        emission_inputs = [eval_z_q]
        if self.params['emission_type'] == 'conditional':
            emission_inputs.append(X)
        if self.params['data_type'] == 'binary_nade':
            self.emission_fxn = theano.function(
                emission_inputs, eval_obs_params[1], name='Emission Function')
        else:
            # eval_obs_params is actually the mean of the output distribution, not a random sample
            self.emission_fxn = theano.function(
                emission_inputs, eval_obs_params[0], name='Emission Function')
        self.posterior_inference = theano.function(
            fxn_inputs, [eval_z_q, eval_mu_q, eval_logcov_q],
            name='Posterior Inference')

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

    def sample(self, nsamples=100, T=10, U=None):
        assert T > 1, 'Sample atleast 2 timesteps'
        # Initial sample
        # XXX: change this to be conditional sample, if necessary
        z = np.random.randn(
            nsamples, 1, self.params['dim_stochastic']).astype(config.floatX)
        assert not np.any(np.isnan(z))
        all_zs = [np.copy(z)]
        assert (U is not None) == bool(self.params['use_cond']), \
            "need U iff use_cond given"
        if U is not None:
            # U assumed to be [T, D]. Same U matrix is used for each generated
            # set of samples.
            dim_cond = self.params['dim_cond']
            expected_shape = (T, dim_cond)
            assert U.shape == expected_shape, \
                "wrong U shape %r, should be %r" % (U.shape, expected_shape)
            U = U.astype(config.floatX)
            assert not np.any(np.isnan(U))
        for t in range(T - 1):
            if U is not None:
                U_vec = U[t].reshape((1, -1))
                stacked = np.stack([U_vec] * nsamples, axis=0)
                # z is N*1*(dim_stochastic), so I'm making this N*1*(dim_cond)
                assert stacked.shape == (nsamples, 1, dim_cond), stacked.shape
                assert not np.any(np.isnan(stacked))
                mu, logcov = self.transition_fxn(z, stacked)
            else:
                mu, logcov = self.transition_fxn(z)
            assert not np.any(np.isnan(mu))
            assert not np.any(np.isnan(logcov))
            z = sampleGaussian(mu, logcov).astype(config.floatX)
            assert not np.any(np.isnan(z))
            all_zs.append(np.copy(z))
        zvec = np.concatenate(all_zs, axis=1)
        if self.params['emission_type'] != 'conditional':
            # easy case: independently sample output at each time step
            X = self.emission_fxn(zvec)
            assert not np.any(np.isnan(X))
        else:
            # hard case: need to collect x at each time step so that we can
            # pass it back in to conditional predictor
            obs_dims = self.params['dim_observations']
            X = np.zeros((nsamples, T, obs_dims), dtype=config.floatX)
            for t in range(T):
                if t == 0:
                    X_in = np.zeros(
                        (nsamples, 1, obs_dims), dtype=config.floatX)
                else:
                    X_in = X[:, t - 1:t]
                z = zvec[:, t:t + 1]
                X[:, t:t + 1] = self.emission_fxn(z, X_in)
                assert not np.any(np.isnan(X))
        return X, zvec

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#


if __name__ == '__main__':
    """ use this to check compilation for various options"""
    from parse_args_dkf import parse
    params = parse()
    if params['use_nade']:
        params['data_type'] = 'binary_nade'
    else:
        params['data_type'] = 'binary'
    params['dim_observations'] = 10
    dkf = DKF(params, paramFile='tmp')
    os.unlink('tmp')
    import ipdb
    ipdb.set_trace()
