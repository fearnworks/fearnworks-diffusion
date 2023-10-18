import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    """
    A variational autoencoder (VAE) encoder that maps an input image to a latent space.
    """
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, Height, Width)
            nn.Conv2d(in_channels=3,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.Conv2d(128,128,kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128,256), # Increase the number of features
            VAE_ResidualBlock(256,256),
            nn.Conv2d(256,256,kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256,512), # Increase the number of features
            VAE_ResidualBlock(512,512),
            nn.Conv2d(512,512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512,512), 
            VAE_ResidualBlock(512,512), 
            VAE_ResidualBlock(256,512), 
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            nn.GroupNorm(32,512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8,8,kernel_size=1, padding=0)
            
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Apply the VAE encoder to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).
            noise (torch.Tensor): The noise tensor of shape (batch_size, out_channels, height/8, width/8).

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, 8, height/8, width/8).
        """        
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0,1,0,1))
            x = F(module(x))
        # (Batch_Size, 8, Height, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4 , Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x,2, dim=1)
        
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        # Z = N(0,1) -> N(mean, variance)=X?
        # X = mean + stdev * Z
        x = mean + stdev * noise
        # Scale the output by a constant
        x *= 0.18215 # Stable    diffusion scale factor https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml#L17
        return x 
        
                
        