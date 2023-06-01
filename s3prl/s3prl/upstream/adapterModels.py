from typing import List
import math

import torch
from torch import nn
# torch.autograd.set_detect_anomaly(True)
'''
from rational.torch import Rational


from .configuration import (
    AdapterFusionConfig,
    AdapterSwitchConfig
)
'''
import logging
from utility.helper import is_leader_process
from dataclasses import dataclass, field
from argparse import Namespace
'''
class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        elif hidden_act.lower() == 'identity':
            self.f = lambda x: x
        elif hidden_act.lower() == 'one':
            def one(x):
                return torch.ones_like(x)
            self.f = one
        elif hidden_act.lower() == 'sigmoid':
            self.f = torch.sigmoid
        elif hidden_act.lower().startswith('rational:'):
            func_name = hidden_act.lower().split(':', 1)[1]
            self.f = Rational(
                cuda=True, trainable=True, train_numerator=True,
                train_denominator=True, version="A", approx_func=func_name
            )

    def forward(self, x):
        return self.f(x)


# Single Adapter


class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
        self,
        input_size,
        down_sample=None,
        non_linearity="relu",
        init_bert_weights=True,
        add_layer_norm_before=True,
        add_layer_norm_after=False,
        residual_before_ln=True,
        skip_linear_layers=False,
        drop_skip_connections=False,
        drop_skip_connections_training_only=False
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln
        self.skip_linear_layers = skip_linear_layers
        self.drop_skip_connections = drop_skip_connections
        self.drop_skip_connections_training_only = drop_skip_connections_training_only

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        if not self.skip_linear_layers:

            # if a downsample size is not passed, we just half the size of the original input
            self.down_sample = down_sample

            if self.down_sample is None:
                self.down_sample = self.input_size // 2

            # Linear down projection of the input
            seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample.
        # In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        if not self.skip_linear_layers:
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)

            # If we want to have a layer norm on output, we apply it later
            # after a separate residual connection
            # This means that we learn a new output layer norm, which replaces
            # another layer norm learned in the bert layer
            if self.add_layer_norm_after:
                self.adapter_norm_after = nn.LayerNorm(self.input_size)
        else:
            self.adapter_up = nn.Identity()

        # if we want to initialize with the bert strategy then this
        # function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, residual_input):  # , residual_input=None):
        down = self.adapter_down(x)
        if not self.skip_linear_layers:
            up = self.adapter_up(down)
            output = up
        else:
            up = down
            output = down

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            if not self.drop_skip_connections or \
               (not self.training and self.drop_skip_connections_training_only):
                output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.residual_before_ln:
            if not self.drop_skip_connections or \
               (not self.training and self.drop_skip_connections_training_only):
                output = output + residual_input

        return output, down, up

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# Adapter Fusion


class BertFusion(nn.Module):
    """
    Implementation of an AdapterFusion block.
    """

    def __init__(
        self,
        config: AdapterFusionConfig,
        dense_size,
        attention_probs_dropout_prob,
    ):
        super(BertFusion, self).__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config

        self.dense_size = dense_size
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if not self.config["query"] and not self.config["key"] and not self.config["value"]:
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config["query"]:
            self.query = nn.Linear(self.dense_size, self.dense_size)
            self.query.apply(Adapter.init_bert_weights)

        if self.config["key"]:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(Adapter.init_bert_weights)

        if self.config["value"]:
            self.value = nn.Linear(self.dense_size, self.dense_size, bias=False)
            self.value.apply(Adapter.init_bert_weights)
            if self.config["value_initialized"]:
                self.value.weight.data = (torch.zeros(self.dense_size, self.dense_size) + 0.000001).fill_diagonal_(1.0)

        if self.config["temperature"]:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual):

        if self.config["residual_before"]:
            value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)

        if self.config["query"]:
            query_layer = self.query(query)
        else:
            query_layer = query

        if self.config["key"]:
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config["value"] and self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2)

        attention_scores = self.dropout(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        if not self.training:
            self.recent_attention = attention_probs.detach().cpu().numpy()

        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)

        if self.config["value"] and not self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer

        if not self.config["residual_before"]:
            context_layer += residual

        return context_layer


# Invertible Adapters

def get_subnet_constructor(non_linearity, reduction_factor):
    def subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, dims_in // reduction_factor),
            Activation_Function_Class(non_linearity),
            nn.Linear(dims_in // reduction_factor, dims_out),
        )

    return subnet


class NICECouplingBlock(nn.Module):
    """Coupling Block following the NICE design."""

    def __init__(self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2):
        super().__init__()

        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        assert all(
            [dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]
        ), "Dimensions of input and one or more conditions don't agree."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.F = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])
        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)

        return torch.cat((y1, y2), -1)

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class GLOWCouplingBlock(nn.Module):
    """
    Coupling Block following the GLOW design. The only difference to the RealNVP coupling blocks, is the fact that it
    uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate subnetworks. This reduces
    computational cost and speeds up learning. clamp: Soft clamping for the multiplicative component. The amplification
    or attenuation of each input dimension can be at most ±exp(clamp).
    """

    def __init__(self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2, clamp=5.0):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = math.exp(clamp)
        self.min_s = math.exp(-clamp)

        assert all(
            [tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]
        ), f"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2 * 2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1 * 2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])

        if not rev:
            s2, t2 = x1.clone(), x2.clone()
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) + torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )

        else:  # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1 :]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = -torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) - torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims
'''

import json
   
# declaringa a class
class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
      
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

def is_baseline(baseline):
    if(type(baseline) == list):
        assert(len(baseline) == 12)
        return baseline
    else:
        if baseline < 0:
            return 0
        return [baseline] * 12
    

class MyLogger:
    def __init__(self, logger) -> None:
        
        self.logger = logger
        
    def info(self, msg):
        if is_leader_process():
            self.logger.info(msg)
logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)
# logger = MyLogger(logger)
@dataclass
class AdapterConfig:
    adapterType: str = field(
        default = "houlsby", metadata={"help": "Adapter type to use."}
    )
    switch: bool = field(
        default = False, metadata={"help": "Whether to use adapter switch."}
    )
    nasPath: int = field(
        default = 2, metadata={"help": "Number of paths in the adapter switch."}
    )
    temperature: float = field(
        default = 0.1, metadata={"help": "Temperature for adapter switch."}
    )
    tauType: str = field(
        default = 'constant', metadata={"help": "Type of tau to use."}
    )
    

class AdapterSwitch(nn.Module):

    config: Namespace

    mode: str

    recent_weights = None

    fixed: int = None

    fixed_idx: int = None

    layer_idx: int = None
    
    hard: bool = False
    
    def __init__(
        self,
        config: object = None,
        initial_logits: List[float] = None,
        layer_idx: int = None
    ):
        super().__init__()

        self.config = config

        # Keep the logits of probabilities as a separate parameters.
        
        self.tau_setup()
        self.switch_temperature = ([self.config.tau.init_value - self.tau_step * self.config.tau.init_steps])
        self.hard = self.config.hard
        # Distribution used.
        self.gumbel = torch.distributions.Gumbel(0, 1)
        self.training = True
        self.paths = self.config.path
        self.layer_idx = layer_idx
        initial_logits = ([1. / len(self.paths)] * len(self.paths) if not (self.config.baseline) 
                            else [int(i == self.config.baseline[layer_idx]) for i in range(len(self.paths))])
        print(initial_logits)
        self.register_parameter(
                    'switch_logits', nn.Parameter(torch.FloatTensor(initial_logits))
                )
        # self.soft_logits = self.probs()
        logger.info(f"paths = {len(initial_logits)}")
        self.prev_mode = None
        self.fixed_idx = len(initial_logits)
        
    @property
    def probs(self):
        return torch.softmax(self.switch_logits / self.switch_temperature[0], dim=-1)

    def get_arch(self):
        return torch.argmax(self.switch_logits, dim=-1).item()
    
    def train(self, mode: bool = True):
        logger.info(f'{"train" if mode else "eval"} invoked')
        # self.switch_logits.requires_grad = mode
        self.training = (mode and self.config.stage != 2)
        if not self.training:
            self.fixed_idx = self.get_arch()
            # self.soft_logits = torch.softmax(self.switch_logits / self.switch_temperature[0], -1)
            logger.info(f'path index after layer_{self.layer_idx}: {self.fixed_idx}')
        else:
            pass
            # self.fixed_idx = None
        return super().train(mode)

    def switch_mode(self):
        if self.prev_mode and not self.switch_logits.requires_grad:
            self.fixed_idx = self.get_arch()
            # logging.warning(f"triggered {self.fixed_idx}")
        
        self.prev_mode = self.switch_logits.requires_grad
        return self.prev_mode
            

    '''
    def eval(self, mode: bool = False):
        self.training = mode
        # logger.info(f"eval invoked, mode = {mode}")
        # self.switch_logits.requires_grad = mode
        if not mode:
            self.fixed_idx = torch.argmax(self.switch_logits, dim=-1).item()
        else:
            self.fixed_idx = None
        return super().eval(mode)
    '''
    def forward(self, x):
        x = x.transpose(0, 1)
        batch_size, seq_length, num_classes, hidden_dim_size = x.size()
        # print('477', self.switch_logits)
        self.switch_mode()
        if not self.training or self.config.stage == 2 or (self.fixed_idx < len(self.switch_logits) and self.probs[self.fixed_idx] > self.config.fix_thres):
            assert(self.fixed_idx < len(self.switch_logits))
            self.switch_logits.requires_grad = False
            # logger.info(f'{x.shape},  {self.fixed_idx} {x[:, :, self.fixed_idx, :].shape}')
            return x[:, :, self.fixed_idx, :].transpose(0, 1)

        if self.config.strategy == 'global':
            sample_size = [batch_size, num_classes]
        elif self.config.strategy == 'seq_length':
            sample_size = [batch_size, seq_length, num_classes]
        else:
            sample_size = [batch_size, seq_length, hidden_dim_size, num_classes]

        # Sample from Gumbel.

        # self.reduce_tau()

        g = self.gumbel.sample(sample_size).to(self.probs.device)

        # Compute the weights of the convex sum.
        weights = torch.softmax((g + self.switch_logits) / self.switch_temperature[0], dim=-1)
        # weights = Gumbel.gumbel_softmax(self.switch_logits, temperature=self.switch_temperature, hard=(not self.training), shape=sample_size)
        
        if self.hard and (self.switch_logits.requires_grad or not self.config.soft_adapter):
            y_hard = Gumbel.onehot_from_logits(weights)
            #print(y_hard[0], "random")
            weights = (y_hard - weights).detach() + weights
        
        # weights = torch.nn.functional.gumbel_softmax(logits=self.switch_logits.expand(sample_size).to(self.probs.device), tau=self.switch_temperature, hard=(self.hard))
        # Compute the output.
        if self.config.strategy == 'global':
            y = torch.einsum('ijkl,ik->ijl', x, weights)
        elif self.config.strategy == 'seq_length':
            y = torch.einsum('ijkl,ijk->ijl', x, weights)
        else:
            y = torch.einsum('ijkl,ijlk->ijl', x, weights)

        # logger.info(f"{y[0]}\n{y.shape}\n{weights}")
        # print('458', x.shape, y.shape, y.transpose(0, 1).shape)
        return y.transpose(0, 1)
    def reduce_tau(self):
        # tau_before = self.switch_temperature[0]
        if self.config.tau.type == 'linear':
            self.switch_temperature[0] = max(self.config.tau.stop_value, self.switch_temperature[0] - self.tau_step)
        elif self.config.tau.type == 'exp':
            self.switch_temperature[0] = max(self.config.tau.stop_value, self.switch_temperature[0] * self.tau_step)
        # logger.info(f"tau reduce from {tau_before} to {self.switch_temperature[0]}")

    def tau_setup(self):
        if self.config.tau.type == 'const':
            self.tau_step = 0
            return
        
        if self.config.tau.type == 'linear':
            if self.config.baseline:
                self.tau_step = 0.0
            else:
                self.tau_step = (not self.config.baseline) * (self.config.tau.init_value - self.config.tau.stop_value) / self.config.tau.steps
            return
        
        if self.config.tau.type == 'exp':
            # init_value * (tau_step) ^ steps = stop_value -> tau_step = (stop_value / init_value) ^ (1 / steps)
            if (self.config.baseline):
                self.tau_step = 1.0
            else:
                self.tau_step = (self.config.tau.stop_value / self.config.tau.init_value) ** (1 / self.config.tau.steps)
            return
        '''
        self.tau_step = 0 if (self.config.baseline) \
                        else (self.config.tau.type == 'linear') * (self.config.tau.init_value - self.config.tau.stop_value) / self.config.tau.steps
        '''
class Adapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class Gumbel:
    def __init__(self) -> None:
        shape = None
    
    @staticmethod
    def onehot_from_logits(logits, eps=0.0):
        """
        Given batch of logits, return one-hot sample using epsilon greedy strategy
        (based on given epsilon)
        """
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        #print(logits[0],"a")
        #print(len(argmax_acs),argmax_acs[0])
        if eps == 0.0:
            return argmax_acs

    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    @staticmethod
    def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
        """Sample from Gumbel(0, 1)"""
        U = torch.nn.Variable(tens_type(*shape).uniform_(), requires_grad=False)
        return -torch.log(-torch.log(U + eps) + eps)

    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    @staticmethod
    def gumbel_softmax_sample(logits, temperature, shape):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + Gumbel.sample_gumbel(logits.shape, tens_type=type(logits.data))
        return torch.nn.functional.softmax(y / temperature, dim=1)

    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    @staticmethod
    def gumbel_softmax(logits, temperature=1.0, hard=False, shape=None):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = Gumbel.gumbel_softmax_sample(logits, temperature, shape)
        if hard:
            y_hard = Gumbel.onehot_from_logits(y)
            #print(y_hard[0], "random")
            y = (y_hard - y).detach() + y
        return y