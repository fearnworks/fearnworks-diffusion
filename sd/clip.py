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
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((num_tokens, dim_embed)))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
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
        
        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(dim_embed)
        # Self attention
        self.attention = SelfAttention(num_heads, dim_embed)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(dim_embed)
        # Feedforward layer
        self.linear_1 = nn.Linear(dim_embed, 4 * dim_embed)
        self.linear_2 = nn.Linear(4 * dim_embed, dim_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the CLIPLayer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, dim_embed).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, seq_len, dim_embed).
        """
        # (Batch_Size, Seq_Len, Dim)
        residue = x
        
        ### SELF ATTENTION ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x

class CLIP(nn.Module):
    """
    A Contrastive Language-Image Pre-Training (CLIP) model that maps text and images to a joint embedding space.
    """
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output