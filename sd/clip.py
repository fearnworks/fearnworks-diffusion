import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    """
    An embedding layer in the Contrastive Language-Image Pre-Training (CLIP) model that maps input tokens to a joint embedding space.

    Args:
        num_vocab (int): The number of tokens in the vocabulary.
        dim_embed (int): The dimensionality of the embedding space.
        num_tokens (int): The maximum number of tokens in a sequence.
    """
    def __init__(self, num_vocab: int, dim_embed: int, num_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(num_vocab, dim_embed)
        self.position_embedding = nn.Parameter(torch.zeros(num_tokens, dim_embed))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:  
        """
        Apply the CLIPEmbedding layer to the input tensor.

        Args:
            tokens (torch.LongTensor): The input tensor of shape (batch_size, seq_len).

        Returns:
            output (torch.FloatTensor): The output tensor of shape (batch_size, seq_len, dim_embed).
        """
        x = self.token_embedding(tokens)
        x += self.position_embedding 
        return x 
    
class CLIPLayer(nn.Module):
    """
    A layer in the Contrastive Language-Image Pre-Training (CLIP) model that applies self-attention and feed-forward layers to an input tensor.
    Args:
        num_heads (int): The number of attention heads to use.
        dim_embed (int): The dimensionality of the embedding space.

    """
    def __init__(self, num_heads: int, dim_embed: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_embed)
        self.attention = SelfAttention(num_heads, dim_embed)
        self.norm2 = nn.LayerNorm(dim_embed)
        self.linear1 = nn.Linear(dim_embed, dim_embed * 4)
        self.linear2 = nn.Linear(dim_embed * 4, dim_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the CLIPLayer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, dim_embed).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, seq_len, dim_embed).
        """
        # (Batch_size, Seq_Len, Dim)
        residue = x 
        x = self.norm1(x)
        x = self.attention(x, causal_mask=True)
        x += x
        
        # Feed Forward
        residue = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = x * torch.signmoid(1.702 * x) # Quick approximation of GELU
        x = self.linear2(x)
        x += residue
        return x

class CLIP(nn.Module):
    """
    A Contrastive Language-Image Pre-Training (CLIP) model that maps text and images to a joint embedding space.
    """
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768. 77)
        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12),
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Apply the CLIP model to the input tensor.

        Args:
            tokens (torch.LongTensor): The input tensor of shape (batch_size, seq_len).

        Returns:
            output (torch.FloatTensor): The output tensor of shape (batch_size, 768).
        """
        tokens = tokens.type(torch.long)
        # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
            
        output = self.layernorm(state)
        return output
        