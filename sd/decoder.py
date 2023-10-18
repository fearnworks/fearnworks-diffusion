import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    """
    A residual block used in the decoder of a variational autoencoder (VAE) that applies
    self-attention to the input tensor.

    Args:
        channels (int): The number of input channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Apply the VAE attention block to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, channels, height, width).
        """
        residue = x
        n,c,h,w = x.shape
        x = x.view(n,c,h*w)
        x = x.transpose(-1,-2) # Move features to end of tensor
        x = self.attention(x)
        x = x.transpose(-1,-2) # Move features back
        x = x.view((n,c,h,w))
        x += residue
        return x 
    
    
class VAE_ResidualBlock(nn.Module):
    """
    A residual block used in the decoder of a variational autoencoder (VAE). 

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the VAE residual block to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, out_channels, height, width).
        """
        residue = x 
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)
        
class VAE_Decoder(nn.Sequential):
    """
    A variational autoencoder (VAE) decoder that maps a latent space to an output image.
    """   
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4, kernel_size=1, padding=0),
            nn.Conv2d(4,512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(256,512),
            nn.Upsample(scale_factor=2), # (Height & Width / 8) -> (Height & Width / 4)
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            nn.Upsample(scale_factor=2), # (Height & Width / 4) -> (Height & Width / 2)
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            nn.Upsample(scale_factor=2), # (Height & Width / 2) -> (Height & Width)
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.GroupNorm(32,128),
            nn.SiLU(),
            nn.Conv2d(128,3, kernel_size=3, padding=1),
            
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the VAE decoder to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 4, height/8, width/8).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, 3, height, width).
        """
        x /= 0.18215
        for module in self:
            x = module(x)
        return x 