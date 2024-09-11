import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Group normalization on input features
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        # First convolution layer
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Group normalization after merging features and time embedding
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        # Second convolution layer
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection layer, either identity or convolution to match channel dimensions
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature):
        # Input feature: (Batch_Size, In_Channels, Height, Width)
        residue = self.residual_layer(feature)  # Store the initial input for the residual path

        # Apply GroupNorm and activation to the feature map
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)

        # Pass through the first convolution layer
        feature = self.conv_feature(feature)

        # Merge feature and residue
        merged = feature + residue

        # Apply GroupNorm, activation, and the second convolution layer
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        # Final output with skip connection from the original input
        return merged + residue

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=256):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = SwitchSequential(
            # (Batch_Size, 128, Height / 64, Width / 64) -> (Batch_Size, 128, Height / 64, Width / 64)
            UNET_AttentionBlock(2, 32), 
            
            # (Batch_Size, 128, Height / 64, Width / 64) -> (Batch_Size, 128, Height / 64, Width / 64)
            UNET_ResidualBlock(64, 128), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 192, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(128, 128), UNET_AttentionBlock(4, 32), Upsample(128), nn.Dropout(0.2)),
            
            # (Batch_Size, 192, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(128, 256), UNET_AttentionBlock(4, 64), Upsample(256), nn.Dropout(0.2)),

            # (Batch_Size, 192, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(256, 256), UNET_AttentionBlock(4, 64), Upsample(256), nn.Dropout(0.2)),

            # (Batch_Size, 192, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(256, 128), UNET_AttentionBlock(4, 32), Upsample(128), nn.Dropout(0.2)),

            # (Batch_Size, 192, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(128, 64), UNET_AttentionBlock(4, 16), Upsample(64), nn.Dropout(0.2)),
            
        ])

    def forward(self, x, context):
        # x: (Batch_Size, 128, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 128)

        x = self.bottleneck(x, context)

        for layers in self.decoders:
            x = layers(x, context)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(2, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 32, Height / 8, Width / 8)

        # (Batch_Size, 32, Height / 8, Width / 8) -> (Batch_Size, 32, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 32, Height / 8, Width / 8) -> (Batch_Size, 32, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 32, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x

class Diffusion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.unet = UNET().to(device)  # Move UNET to the device
        self.final = UNET_OutputLayer(64, 3).to(device)  # Move UNET_OutputLayer to the device
        self.linear_layer_context = nn.Linear(512, 77 * 256).to(device)  # Move linear layer to the device
        self.linear_layer_img = nn.Linear(512, 7*7*64).to(device)  # Move linear layer to the device
    
    def forward(self, latent, context):
        # Ensure inputs are on the same device
        latent = latent.to(self.device)
        context = context.to(self.device)
        latent = self.linear_layer_img(context)
        # Transform the latent tensor
        latent = latent.view(latent.size(0), 64, 7, 7)
        
        # Transform the context tensor using the pre-initialized linear layer
        context = self.linear_layer_context(context)
        context = context.view(context.size(0), 77, 256)

        # Pass through UNET
        output = self.unet(latent, context)
        
        # Optional: Pass through final output layer if needed
        output = self.final(output)
        
        return output