import torch
from torch import nn 
from torch.nn import functional as F 
import math

class SelfAttention(nn.Module):
    """
    A self-attention module that applies multi-head attention to an input tensor.

    Args:
        num_heads (int): The number of attention heads to use.
        dim_embed (int): The dimensionality of the input embedding.
        in_proj_bias (bool): Whether to include bias terms in the input projection layer.
        out_proj_bias (bool): Whether to include bias terms in the output projection layer.
    """
    def __init__(self, num_heads: int, dim_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(dim_embed, 3 * dim_embed, bias=in_proj_bias)
        self.out_project = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
        self.num_heads = num_heads
        self.dim_head = dim_embed // num_heads
        
    def forward(self, x: torch.Tensor, causal_mask=False):
        """
        Apply multi-head self-attention to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, dim_embed).
            causal_mask (bool): Whether to apply a causal mask to the attention weights.

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, sequence_length, dim_embed).
        """
        
        input_shape = x.shape
        batch_size, sequence_length, dim_embed = input_shape
        intermediate_shape = (batch_size, sequence_length, self.num_heads, self.dim_head)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensors of shape (Batch-size, Seq_Len, Dim)
        q,k,v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(intermediate_shape).transpose(1,2)
        k = k.view(intermediate_shape).transpose(1,2)
        v = v.view(intermediate_shape).transpose(1,2)
        
        weight = q @ k.transpose(-1,-2)
        
        if causal_mask:
            # mask where the upper triangle (above the diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.dim_head)
        
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v 
        # (Batch_size, H, Seq_len, Dim / H ) -> (Batch_Size, Seq_Len, H, Dim/H)
        output = output.transpose(1,2)
        output = output.reshape(input_shape)
        output = self.out_project(output)
        return output
        
        