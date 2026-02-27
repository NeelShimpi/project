import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding using convolution
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (batch_size, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, num_patches, embed_dim * 3)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', 
                       three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, heads, num_patches, num_patches)
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)  # (batch_size, heads, num_patches, head_dim)
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        
        # Final projection
        output = self.projection(attention_output)
        output = self.dropout(output)
        
        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network)
    """
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block with Multi-Head Attention and MLP
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Multi-Head Attention with residual connection
        attention_output = self.attention(self.layer_norm1(x))
        x = x + attention_output
        
        # MLP with residual connection
        mlp_output = self.mlp(self.layer_norm2(x))
        x = x + mlp_output
        
        return x


class VisionTransformerDrowsiness(nn.Module):
    """
    Vision Transformer for Driver Drowsiness Detection
    
    Args:
        img_size: Input image size (default: 224)
        patch_size: Size of image patches (default: 16)
        in_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 2 - alert/drowsy)
        dim: Embedding dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        heads: Number of attention heads (default: 12)
        mlp_dim: Hidden dimension of MLP (default: 3072)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 num_classes=2,
                 dim=768, 
                 depth=12, 
                 heads=12, 
                 mlp_dim=3072, 
                 dropout=0.1):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embedding.num_patches
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.layer_norm = nn.LayerNorm(dim)
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, num_patches, dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches + 1, dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer encoder
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        
        # Classification using class token
        cls_output = x[:, 0]  # Take the class token
        logits = self.mlp_head(cls_output)
        
        return logits
    
    def get_attention_maps(self, x):
        """
        Get attention maps for visualization
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        attention_maps = []
        
        # Get attention weights from each block
        for block in self.transformer_blocks:
            # We need to modify the attention block to return attention weights
            # For now, just pass through
            x = block(x)
        
        return attention_maps


# Lighter version for faster inference
class VisionTransformerDrowsinessLight(nn.Module):
    """
    Lightweight Vision Transformer for real-time drowsiness detection
    Reduced parameters for faster inference on edge devices
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 num_classes=2,
                 dim=384,  # Reduced from 768
                 depth=6,  # Reduced from 12
                 heads=6,  # Reduced from 12
                 mlp_dim=1536,  # Reduced from 3072
                 dropout=0.1):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embedding.num_patches
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.layer_norm = nn.LayerNorm(dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        cls_output = x[:, 0]
        logits = self.mlp_head(cls_output)
        
        return logits


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Full model
    model = VisionTransformerDrowsiness().to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    print(f"Full ViT output shape: {output.shape}")
    print(f"Full ViT parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Light model
    model_light = VisionTransformerDrowsinessLight().to(device)
    output_light = model_light(x)
    print(f"\nLight ViT output shape: {output_light.shape}")
    print(f"Light ViT parameters: {sum(p.numel() for p in model_light.parameters()):,}")
