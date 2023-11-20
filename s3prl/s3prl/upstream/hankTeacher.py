import torch
from torch import nn

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
    def __init__(self, fatherModule: nn.Module):
        super().__init__()
        for name, module in fatherModule.named_children():
            self.add_module(name, module)
        
        for name, module in self.named_parameters():
            if 'bias' in name:
                parent, lastKey, child = find_module(self, name)
                parent.register_parameter(f"bitfit_{lastKey}", nn.Parameter(torch.zeros_like(child)))
                
    def forward(self, x, **kwargs):
        self_attn_mask = kwargs['self_attn_mask']
        self_attn_padding_mask = kwargs['self,_attn_padding_mask']
        need_weights = kwargs['need_weights']
        att_args = kwargs['att_args']
        # Start forward
        residual = x
        lora_out, lora_attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            use_bitfit=True,
        )

        lora_out = self.drouput1(lora_out)
        lora_out = residual + lora_out
        lora_out = self.self_attn_layer_norm(lora_out)

        residual = lora_out
        lora_out = self.activation_fn(self.fc1(lora_out) + self.fc1.bitfit_bias)
        lora_out = self.dropout2(lora_out)

        lora_out = self.fc2(lora_out) + self.fc2.bitfit_bias
        layer_result = lora_out
        lora_out = self.dropout3(lora_out)

        lora_out = lora_out + residual

        lora_out = self.final_layer_norm(lora_out)

        return lora_out, (lora_attn, layer_result)
    
    
