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
import loralib as lora
import torch.nn.functional as F
def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
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
    logging.warning(type(baseline))

    if(type(baseline) == list):
        assert(len(baseline) == 12)
        if type(baseline[0]) is not list:
            for i, b in enumerate(baseline):
                baseline[i] = [b]
        return baseline
    else:
        if baseline < 0:
            return 0
        return [[baseline] for _ in range(12)]
    

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
        layer_idx: int = None,
        all_adapter_type: List[str] = None,
        used_adapter_name: List[str] = None
    ):
        super().__init__()

        self.config = config
        # Keep the logits of probabilities as a separate parameters.
        
        self.tau_setup()
        self.switch_temperature = ([self.config.tau.init_value - self.tau_step * self.config.tau.init_steps])
        # Distribution used.
        self.gumbel = torch.distributions.Gumbel(0, 1)
        self.training = True
        self.paths = self.config.path
        self.layer_idx = layer_idx

        if self.config.algo.name not in ['fair_darts', 's3delta']:
            initial_logits = ([1. / len(self.paths)] * len(self.paths) if not (self.config.baseline) 
                            else [int(i == self.config.baseline[layer_idx]) for i in range(len(self.paths))])
        elif self.config.algo.name == 'fair_darts':
            if not self.config.baseline:
                initial_logits = [0. for _ in range(len(self.paths))]
                self.initial_logits = torch.sigmoid(torch.FloatTensor(initial_logits))
            else:
                initial_logits = [int(i == self.config.baseline[layer_idx]) for i in range(len(self.paths))]
        else:
            if not self.config.baseline:
                initial_logits = [0. for _ in range(len(self.paths))]
                self.initial_logits = torch.sigmoid(torch.FloatTensor(initial_logits))
            else:
                initial_logits = [int(all_adapter_type.index(adapter_name) in self.config.baseline[layer_idx]) for adapter_name in used_adapter_name]

        self.register_parameter(
                    'switch_logits', nn.Parameter(torch.FloatTensor(initial_logits))
                )

        if not self.config.baseline:
            logger.info(f'initial_logits: {torch.sigmoid(self.switch_logits) if self.config.algo.name in ["fair_darts", "s3delta"] else self.switch_logits}')
        else:
            logger.info(f'initial_logits: {self.switch_logits}')

        logger.info(f"paths = {len(initial_logits)}")

        self.prev_mode = None
        self.fixed_idx = None if not self.config.baseline else self.paths
        self.used_adapter = used_adapter_name
        # For algorithms sampling noise from Gumbel distribution
        if self.config.algo.use_gumbel:
            self.gumbel_noise = None
        if self.config.algo.name == 's3delta':
            self.uniform_noise = None
            self.p = torch.zeros_like(self.switch_logits)

    @property
    def probs(self):
        if self.config.algo.name == 'fair_darts':
            return torch.sigmoid(self.switch_logits)
        elif self.config.algo.name == 's3delta':
            return self.p
        else:
            return torch.softmax(self.switch_logits / self.switch_temperature[0], dim=-1)

    def get_arch(self):
        return [torch.argmax(self.switch_logits, dim=-1).item()]
    
    def train(self, mode: bool = True):
        logger.info(f'{"train" if mode else "eval"} invoked')
        # self.switch_logits.requires_grad = mode
        self.training = (mode and self.config.stage != 2)
        if not self.training:
            if self.config.algo.name != 's3delta':
                self.fixed_idx = self.get_arch()    # s3delta will call set_hard_forward_structure() in upstream_expert.
            if self.fixed_idx:
                logger.info(f'current adapter of layer_{self.layer_idx}: {[self.used_adapter[idx] for idx in self.fixed_idx]}')
            else:
                logger.info(f'There\'s no current adapter of layer_{self.layer_idx}. Use skip by default.')
        else:
            pass
            # self.fixed_idx = None
        if self.config.algo.name in ['fair_darts', 's3delta'] and not self.config.baseline:
            self.initial_logits = self.initial_logits.to(self.switch_logits.device)
        
        return super().train(mode)

    def switch_mode(self):
        if self.prev_mode and not self.switch_logits.requires_grad:
            self.fixed_idx = self.get_arch()
            # logging.warning(f"triggered {self.fixed_idx}")
        
        self.prev_mode = self.switch_logits.requires_grad
        return self.prev_mode
    
    def aux_loss(self):
        if self.config.algo.name == 'fair_darts':
            # Fair-DARTS: zero-one loss = -1/N * sum(sigmoid(alpha) - 0.5) * scaling_factor
            return -F.mse_loss(torch.sigmoid(self.switch_logits), self.initial_logits.to(self.switch_logits.device))
        raise NotImplementedError

    def sample_gumbel(self):
        batch_size = 1
        num_classes = (self.switch_logits.shape)[-1]

        if self.config.strategy == 'global':
            sample_size = [batch_size, num_classes]
        else:
            raise NotImplementedError
        
        self.gumbel_noise =  self.gumbel.sample(sample_size).to(self.probs.device)
    
    def sample_uniform(self):
        self.uniform_noise = torch.rand(len(self.p)).to(self.p.device)

    def forward(self):
        self.switch_mode()

        if self.config.algo.name == 'darts':
            weights = torch.softmax(self.switch_logits, dim=-1).unsqueeze(0)
        elif self.config.algo.name == 'fair_darts':
            weights = torch.sigmoid(self.switch_logits).unsqueeze(0)
        elif self.config.algo.name == 's3delta':
            self.uniform_noise = self.uniform_noise.to(self.p.device)
            # Calculate z_hat = sigmoid(log[(u * p) / {(1 - u) * (1 - p)}] / tau)
            z_hat = torch.sigmoid(
                torch.log((self.uniform_noise * self.p) / ((1 - self.uniform_noise) * (1 - self.p))) / self.switch_temperature[0]
            )
            weights = torch.ones_like(z_hat) - z_hat.detach() + z_hat
            # Close a gate if z_hat_i < 0.5
            unused_index = (z_hat < 0.5).nonzero()
            weights[unused_index] = 0.0
            if torch.sum(weights) == 0.0:
                logger.info(f'layer {self.layer_idx}: none has been chosen, z_hat = {z_hat}, use the path with the largest p')
                max_idx = torch.argmax(self.p)
                weights[max_idx] = 1 - self.p[max_idx].detach() + self.p[max_idx]
            weights = weights.unsqueeze(0)
        elif self.config.algo.name in ['gdas', 'gumbel_darts']:
            # Compute the weights of the convex sum.
            weights = torch.softmax((self.gumbel_noise + self.switch_logits) / self.switch_temperature[0], dim=-1)
            # weights = Gumbel.gumbel_softmax(self.switch_logits, temperature=self.switch_temperature, hard=(not self.training), shape=sample_size)
            if (self.switch_logits.requires_grad and not self.config.algo.soft_switch) or \
                (not self.switch_logits.requires_grad and not self.config.algo.soft_train):
                y_hard = Gumbel.onehot_from_logits(weights)
                weights = (y_hard - weights).detach() + weights
        else:
            raise NotImplementedError(f'{self.config.algo.name} did not implemented!')

        return weights, torch.argmax(weights[0], dim=-1)

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
            if self.config.baseline or self.config.tau.steps == 0:
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
    
class PAdapter(nn.Module):
    def __init__(self, parentModule: nn.Module) -> None:
        super().__init__()
        
        for name, module in parentModule.named_children():
            # self.add_module(name, module)
            pass
            
        emb_dim = parentModule.embedding_dim
        self.adapter = nn.Sequential(
                    nn.Linear(emb_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, emb_dim),
                )
        self.name = 'para'
        self.attn = self.layer_result = None
        # self.activation_fn = parentModule.activation_fn
        self.num_param = sum(p.nelement() for p in self.parameters())
    
    @property
    def num_parameter(self):
        return (self.num_param / 1e6)
    
    def forward(self, x, parent, **kwargs):
        residual = x
        parallel_input = x
        self_attn_mask = kwargs['self_attn_mask']
        self_attn_padding_mask = kwargs['self_attn_padding_mask']
        need_weights = kwargs['need_weights']
        att_args = kwargs['att_args']
        x, self.attn = parent.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
        )

        x = parent.dropout1(x)
        x = residual + x

        x = parent.self_attn_layer_norm(x)

        residual = x
        x = parent.activation_fn(parent.fc1(x))
        #
        x = parent.dropout2(x)
        x = parent.fc2(x)
        #
        self.layer_result = x

        x = parent.dropout3(x)
        adapter_output = self.adapter(parallel_input)
            
        x = x + adapter_output + residual
        # return x
        x = parent.final_layer_norm(x)

        return x, (self.attn, self.layer_result)

class SAdapter(nn.Module):
    def __init__(self, parentModule: nn.Module) -> None:
        super().__init__()
        
        for name, module in parentModule.named_children():
            pass
            # self.add_module(name, module)
            
        emb_dim = parentModule.embedding_dim
        self.adapter = nn.Sequential(
                    nn.Linear(emb_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, emb_dim),
                )
        
        self.name = 'seq'
        self.attn = self.layer_result = None
        # self.activation_fn = parentModule.activation_fn
        self.num_param = sum(p.nelement() for p in self.parameters())
    
    @property
    def num_parameter(self):
        return self.num_param / 1e6
    
    def forward(self, x, parent: nn.Module, **kwargs):
        residual = x
        self_attn_mask = kwargs['self_attn_mask']
        self_attn_padding_mask = kwargs['self_attn_padding_mask']
        need_weights = kwargs['need_weights']
        att_args = kwargs['att_args']
        x, self.attn = parent.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
        )

        x = parent.dropout1(x)
        x = residual + x

        x = parent.self_attn_layer_norm(x)

        residual = x
        x = parent.activation_fn(parent.fc1(x))
        #
        x = parent.dropout2(x)
        x = parent.fc2(x)
        #
        houlsby_input = x
        #
        self.layer_result = x

        x = parent.dropout3(x)
        adapter_output = self.adapter(houlsby_input)
            
        x = x + adapter_output + residual
        # return x
        x = parent.final_layer_norm(x)

        return x, (self.attn, self.layer_result)

# from s3prl.upstream.wav2vec2.wav2vec2_model import quant_noise

class LoRAAdapter(nn.Module):
    # Use LoRA for q_proj and v_proj in MultiHeadAttention
    def __init__(self, parentModule: nn.Module):
        super().__init__()
        for name, module in parentModule.named_children():
            # self.add_module(name, module)
            pass
        
        for name, module in parentModule.named_parameters():
            for loraName in ['q_proj', 'v_proj']:
                if loraName in name and not getattr(module, '__is_delta__', False) and not getattr(module, '__is_virtual__', False):
                    loraKey = name.split(loraName)[0] + loraName
                    parent, lastKey, child = find_module(parentModule, loraKey)
                    if getattr(self, lastKey, False):
                        continue
                    outdim, indim  = child.weight.shape
                    setattr(self, lastKey, quant_noise(lora.Linear(indim, outdim, r=8), parentModule.self_attn.q_noise, parentModule.self_attn.qn_block_size))
                    ref = getattr(self, lastKey, None)
                    ref.weight = child.weight
                    ref.bias = child.bias

        self.name = 'lora'
        self.lora_keys = ['q_proj', 'v_proj']
        self.attn = self.layer_result = None
        self.num_param = sum(p.nelement() for n, p in self.named_parameters() if 'lora' in n)
    
    @property
    def num_parameter(self):
        return self.num_param / 1e6

    def lora_weights(self):
        ref = {key: getattr(self, key, None) for key in self.lora_keys}
        return {
            key: {'lora_A': ref[key].lora_A, 'lora_B': ref[key].lora_B}
            for key in self.lora_keys
        }
    
    def lora_scaling(self):
        ref = {key: getattr(self, key, None) for key in self.lora_keys}
        return {
            key: ref[key].scaling
            for key in self.lora_keys
        }

    def forward(self, x, parent, **kwargs):
        residual = x
        self_attn_mask = kwargs['self_attn_mask']
        self_attn_padding_mask = kwargs['self_attn_padding_mask']
        need_weights = kwargs['need_weights']
        att_args = kwargs['att_args']
        # Start forward
        residual = x
        lora_out, self.attn = parent.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            use_lora=True,
            lora_weights = self.lora_weights(),
            lora_scaling = self.lora_scaling()
        )

        lora_out = parent.dropout1(lora_out)
        lora_out = residual + lora_out
        lora_out = parent.self_attn_layer_norm(lora_out)

        residual = lora_out
        lora_out = parent.activation_fn(parent.fc1(lora_out))
        lora_out = parent.dropout2(lora_out)

        lora_out = parent.fc2(lora_out)
        self.layer_result = lora_out
        lora_out = parent.dropout3(lora_out)

        lora_out = lora_out + residual
        lora_out = parent.final_layer_norm(lora_out)

        return lora_out, (self.attn, self.layer_result)


class LNFitAdapter(nn.Module):
    def __init__(self, parentModule):
        super().__init__()
        for name, module in parentModule.named_children():
            pass
            # self.add_module(name, module)
        
        self.lnfit_self_attn_layer_norm = nn.LayerNorm(parentModule.embedding_dim, eps=1e-5, elementwise_affine=True)
        self.lnfit_final_layer_norm = nn.LayerNorm(parentModule.embedding_dim, eps=1e-5, elementwise_affine=True)

        self.name = 'lnfit'
        self.attn = self.layer_result = None
        self.num_param = sum(p.nelement() for p in self.parameters())
    
    @property
    def num_parameter(self):
        return self.num_param / 1e6
    
    def forward(self, x, parent, **kwargs):
        self_attn_mask = kwargs['self_attn_mask']
        self_attn_padding_mask = kwargs['self_attn_padding_mask']
        need_weights = kwargs['need_weights']
        att_args = kwargs['att_args']
        # Start
        residual = x
        lnfit_out, self.attn = parent.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False
        )

        lnfit_out = parent.dropout1(lnfit_out)
        lnfit_out = residual + lnfit_out
        lnfit_out = self.lnfit_self_attn_layer_norm(lnfit_out)

        residual = lnfit_out
        lnfit_out = parent.activation_fn(parent.fc1(lnfit_out))
        lnfit_out = parent.dropout2(lnfit_out)

        lnfit_out = parent.fc2(lnfit_out)
        self.layer_result = lnfit_out
        lnfit_out = parent.dropout3(lnfit_out)

        lnfit_out = lnfit_out + residual
        lnfit_out = self.lnfit_final_layer_norm(lnfit_out)

        return lnfit_out, (self.attn, self.layer_result)
    

class FuckBit(nn.Module):
    def __init__(self, init_method="zero"):
        super().__init__()
        self.init_method=init_method
        self.instantiated = False
        
    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True
    def forward(self, output):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            # print(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)

        modified_output =  hiddens + self.bias
        
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output

def find_module(root_module: nn.Module, key:str):
    r"""Find the module using a key and the root module. Return both the parent reference, the child name and reference.

    Args:
        root_module (:obj:`root_module`): The root_module to find the sub module in
        key (:obj:`str`): The relative key to the root module.

    Returns:
        (:obj:`nn.Module`, :obj:`str`, :obj:`nn.Module`):
        * A reference to the parent module of the target module, mainly for substuting the target module.
        * The key of the target module relevant to its parent module
        * Target module.
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module



class BitFitAdapter(nn.Module):
    def __init__(self, parentModule: nn.Module):
        super().__init__()
        for name, module in parentModule.named_children():
            pass
            # self.add_module(name, module)
        
        parent_refs = []
        for name, module in parentModule.named_parameters():
            if 'bias' in name and 'lnfit' not in name and not getattr(module, '__is_delta__', False) and not getattr(module, '__is_virtual__', False):
                # name: module.bias
                parent, lastKey, child = find_module(parentModule, name)
                setattr(self, f'{name.split(".")[-2]}_bitfit_bias', nn.Parameter(torch.zeros_like(child)))
        
        self.name = 'bitfit'
        self.layer_result = self.attn = None
        self.num_param = sum(p.nelement() for p in self.parameters())
    
    @property
    def num_parameter(self):
        return self.num_param / 1e6

    def bitfit_weights(self):
        return {
            'q_proj': self.q_proj_bitfit_bias,
            'k_proj': self.k_proj_bitfit_bias,
            'v_proj': self.v_proj_bitfit_bias,
            'out_proj': self.out_proj_bitfit_bias
        }

    def forward(self, x, parent, **kwargs):
        residual = x
        self_attn_mask = kwargs['self_attn_mask']
        self_attn_padding_mask = kwargs['self_attn_padding_mask']
        need_weights = kwargs['need_weights']
        att_args = kwargs['att_args']
        # Start forward
        residual = x
        bitfit_out, self.attn = parent.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            use_bitfit=True,
            bitfit_weights=self.bitfit_weights(),
        )

        bitfit_out = parent.dropout1(bitfit_out)
        bitfit_out = residual + bitfit_out
        bitfit_out = parent.self_attn_layer_norm(bitfit_out) + self.self_attn_layer_norm_bitfit_bias

        residual = bitfit_out
        bitfit_out = parent.activation_fn(parent.fc1(bitfit_out) + self.fc1_bitfit_bias)
        bitfit_out = parent.dropout2(bitfit_out)

        bitfit_out = parent.fc2(bitfit_out) + self.fc2_bitfit_bias
        self.layer_result = bitfit_out
        bitfit_out = parent.dropout3(bitfit_out)

        bitfit_out = bitfit_out + residual
        bitfit_out = parent.final_layer_norm(bitfit_out) + self.final_layer_norm_bitfit_bias

        return bitfit_out, (self.attn, self.layer_result)
    
class Skip(nn.Module):
    def __init__(self, parentModule: nn.Module):
        super().__init__()
        for name, module in parentModule.named_children():
            pass
            # self.add_module(name, module)
        self.activation_fn = parentModule.activation_fn
        self.layer_result = self.attn = None

        self.name = 'skip'
    
    def forward(self, x, parent, **kwargs):
        residual = x
        self_attn_mask = kwargs['self_attn_mask']
        self_attn_padding_mask = kwargs['self_attn_padding_mask']
        need_weights = kwargs['need_weights']
        att_args = kwargs['att_args']
        # Start forward
        residual = x
        skip_out, skip_attn = parent.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
        )

        skip_out = parent.dropout1(skip_out)
        skip_out = residual + skip_out
        skip_out = parent.self_attn_layer_norm(skip_out)

        residual = skip_out
        skip_out = parent.activation_fn(parent.fc1(skip_out))
        skip_out = parent.dropout2(skip_out)

        skip_out = parent.fc2(skip_out)
        layer_result = skip_out
        self.attn = skip_attn
        self.layer_result = layer_result
        skip_out = parent.dropout3(skip_out)

        skip_out = skip_out + residual
        skip_out = parent.final_layer_norm(skip_out)

        return skip_out, (skip_attn, layer_result)