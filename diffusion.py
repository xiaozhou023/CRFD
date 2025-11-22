import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Define nin_shortcut or conv_shortcut if in_channels != out_channels
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        w_ = torch.bmm(q, k)     # b, hw, hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b, hw, hw
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(CrossAttention, self).__init__()
        # Project image features to Query
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        # Project condition features to Key and Value
        self.key_proj = nn.Linear(condition_dim, feature_dim)
        self.value_proj = nn.Linear(condition_dim, feature_dim)
        # For final output transformation
        self.output_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)


    def forward(self, x, condition,scale=1.0):
        # Convolutional projection of image features to generate Query
        query = self.query_proj(x)  # (B, C_feat, H, W)

        # Linear projection of condition features to generate Key and Value
        key = self.key_proj(condition)  # (B, C_feat)
        value = self.value_proj(condition)  # (B, C_feat)

        # Reshape for matrix multiplication
        b, c, h, w = query.shape
        query = query.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C_feat)
        key = key.unsqueeze(1)  # (B, 1, C_feat)
        value = value.unsqueeze(1)  # (B, 1, C_feat)

        # Calculate attention scores, dot product of query and key
        attention_scores = torch.bmm(query, key.permute(0, 2, 1)) * (c ** -0.5)  # (B, H*W, 1)

        # Introduce learnable scale parameter to adjust the magnitude of attention scores
        attention_scores = attention_scores * scale

        # Calculate attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, H*W, 1)

        # Weight the value using attention weights
        attended_features = torch.bmm(attention_weights, value)  # (B, H*W, C_feat)
        attended_features = attended_features.permute(0, 2, 1).view(b, c, h, w)  # (B, C_feat, H, W)

        # Final output through output_proj
        output = self.output_proj(attended_features) + x  # (B, C_feat, H, W)

        # Reshape attention weights to the same shape as the original input
        attention_map = attention_weights.view(b, h, w)

        return output, attention_map

class ConditionalUNet(nn.Module):
    def __init__(self, config, feature_dim=512):
        super(ConditionalUNet, self).__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])
        mask=1
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels+mask, self.ch, kernel_size=3, stride=1, padding=1)

        # Initialize downsampling layers
        self.down = nn.ModuleList()
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            down.cross_attention = CrossAttention(block_out, feature_dim)  # Add cross-attention module
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle layers
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.cross_attention_mid = CrossAttention(block_in, feature_dim)  # Add cross-attention in the middle layer
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # Initialize upsampling layers
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            up.cross_attention = CrossAttention(block_out, feature_dim)  # Add cross-attention module for upsampling
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # End layers
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, face_features, masks,scale=1.0): #main
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # Concatenate input features and masks
        input_with_mask = torch.cat((x, masks), dim=1)
        x = input_with_mask

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                
                # Add cross attention
                h, _ = self.down[i_level].cross_attention(h, face_features,scale)  # Use cross-attention
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(h))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h, _ = self.mid.cross_attention_mid(h, face_features,scale)  # Cross-attention in the middle layer
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                
                # Add cross attention
                h, _ = self.up[i_level].cross_attention(h, face_features,scale)  # Use cross-attention
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # End
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    # def forward(self, x, t, face_features, masks, scale=1.0): #no features
    #     temb = get_timestep_embedding(t, self.ch)
    #     temb = self.temb.dense[0](temb)
    #     temb = nonlinearity(temb)
    #     temb = self.temb.dense[1](temb)

    #     x = torch.cat((x, masks), dim=1)

    #     hs = [self.conv_in(x)]
    #     for i_level in range(self.num_resolutions):
    #         for i_block in range(self.num_res_blocks):
    #             h = self.down[i_level].block[i_block](hs[-1], temb)
    #             if len(self.down[i_level].attn) > 0:
    #                 h = self.down[i_level].attn[i_block](h)
    #             hs.append(h)  # Do not use cross-attention
    #         if i_level != self.num_resolutions - 1:
    #             hs.append(self.down[i_level].downsample(h))

    #     h = hs[-1]
    #     h = self.mid.block_1(h, temb)
    #     h = self.mid.attn_1(h)
    #     h = self.mid.block_2(h, temb)  # Do not use middle cross-attention

    #     for i_level in reversed(range(self.num_resolutions)):
    #         for i_block in range(self.num_res_blocks + 1):
    #             h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
    #             if len(self.up[i_level].attn) > 0:
    #                 h = self.up[i_level].attn[i_block](h)
    #             # Do not use cross-attention
    #         if i_level != 0:
    #             h = self.up[i_level].upsample(h)

    #     h = self.norm_out(h)
    #     h = nonlinearity(h)
    #     h = self.conv_out(h)
    #     return h
    
    # def forward(self, x, t, face_features, masks, scale=1.0): #no masks
    #     temb = get_timestep_embedding(t, self.ch)
    #     temb = self.temb.dense[0](temb)
    #     temb = nonlinearity(temb)
    #     temb = self.temb.dense[1](temb)

    #     # Do not concatenate mask
    #     hs = [self.conv_in(x)]
    #     for i_level in range(self.num_resolutions):
    #         for i_block in range(self.num_res_blocks):
    #             h = self.down[i_level].block[i_block](hs[-1], temb)
    #             if len(self.down[i_level].attn) > 0:
    #                 h = self.down[i_level].attn[i_block](h)
    #             h, _ = self.down[i_level].cross_attention(h, face_features, scale)  # Retain cross-attention
    #             hs.append(h)
    #         if i_level != self.num_resolutions - 1:
    #             hs.append(self.down[i_level].downsample(h))

    #     h = hs[-1]
    #     h = self.mid.block_1(h, temb)
    #     h = self.mid.attn_1(h)
    #     h, _ = self.mid.cross_attention_mid(h, face_features, scale)
    #     h = self.mid.block_2(h, temb)

    #     for i_level in reversed(range(self.num_resolutions)):
    #         for i_block in range(self.num_res_blocks + 1):
    #             h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
    #             if len(self.up[i_level].attn) > 0:
    #                 h = self.up[i_level].attn[i_block](h)
    #             h, _ = self.up[i_level].cross_attention(h, face_features, scale)
    #         if i_level != 0:
    #             h = self.up[i_level].upsample(h)

    #     h = self.norm_out(h)
    #     h = nonlinearity(h)
    #     h = self.conv_out(h)
    #     return h

    # def forward(self, x, t, face_features, masks, scale=1.0): #both not
    #     temb = get_timestep_embedding(t, self.ch)
    #     temb = self.temb.dense[0](temb)
    #     temb = nonlinearity(temb)
    #     temb = self.temb.dense[1](temb)

    #     # Do not concatenate mask
    #     hs = [self.conv_in(x)]
    #     for i_level in range(self.num_resolutions):
    #         for i_block in range(self.num_res_blocks):
    #             h = self.down[i_level].block[i_block](hs[-1], temb)
    #             if len(self.down[i_level].attn) > 0:
    #                 h = self.down[i_level].attn[i_block](h)
    #             hs.append(h)  # Do not use cross-attention
    #         if i_level != self.num_resolutions - 1:
    #             hs.append(self.down[i_level].downsample(h))

    #     h = hs[-1]
    #     h = self.mid.block_1(h, temb)
    #     h = self.mid.attn_1(h)
    #     h = self.mid.block_2(h, temb)

    #     for i_level in reversed(range(self.num_resolutions)):
    #         for i_block in range(self.num_res_blocks + 1):
    #             h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
    #             if len(self.up[i_level].attn) > 0:
    #                 h = self.up[i_level].attn[i_block](h)
    #         if i_level != 0:
    #             h = self.up[i_level].upsample(h)

    #     h = self.norm_out(h)
    #     h = nonlinearity(h)
    #     h = self.conv_out(h)
    #     return h
