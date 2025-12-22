import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 原来的 DoubleConv：保留接口以防外部引用（SeaFormer 不再使用）
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# =========================================================
# SeaFormer 风格模块：Overlap Patch Embedding
# =========================================================
class OverlapPatchEmbedding(nn.Module):
    """
    类似 SegFormer/SeaFormer 的 Overlap Patch Embedding：
    通过带 stride 的卷积做 patch 划分，同时保留部分空间信息。
    """
    def __init__(self, in_channels, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, C', H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', C']
        x = self.norm(x)
        return x, H, W


# =========================================================
# SeaFormer 风格：Squeeze-Enhanced Axial Attention (SEA attention)
# =========================================================
class SEAAttention(nn.Module):
    """
    Squeeze-Enhanced Axial Attention（简化实现版）：
    - 使用 1x1 Conv 生成 q, k, v
    - 沿 H/W 维度做 squeeze（平均池化）以获得轴向特征
    - 在轴向特征上做 self-attention，并通过 broadcast 还原到 2D
    - 通过 3x3 depthwise conv + 1x1 conv 生成细节增强权重，与全局语义特征相乘
    计算复杂度随 HW 线性变化，更接近 SeaFormer 原文设计。
    """
    def __init__(self, dim, attn_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.qk_dim = max(int(dim * attn_ratio), 8)
        self.v_dim = dim

        # 共享的 qkv 生成层（同时给 squeeze 路径和 detail 路径使用）
        self.qkv = nn.Conv2d(dim, self.qk_dim * 2 + self.v_dim, kernel_size=1, bias=False)

        # 细节增强卷积核（depthwise conv + 1x1 conv）
        self.enh_dwconv = nn.Conv2d(
            self.qk_dim * 2 + self.v_dim,
            self.qk_dim * 2 + self.v_dim,
            kernel_size=3,
            padding=1,
            groups=self.qk_dim * 2 + self.v_dim,
            bias=False,
        )
        self.enh_bn1 = nn.BatchNorm2d(self.qk_dim * 2 + self.v_dim)
        self.enh_proj = nn.Conv2d(self.qk_dim * 2 + self.v_dim, dim, kernel_size=1, bias=True)
        self.enh_bn2 = nn.BatchNorm2d(dim)

        self.scale = self.qk_dim ** -0.5

    def forward(self, x):
        """
        x: [B, C, H, W]
        返回: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1) 生成 q, k, v
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.v_dim], dim=1)

        # -------- Squeeze Axial attention：全局语义 --------
        # 水平方向 squeeze，得到 H 维度的轴向特征
        # 形状: [B, Cqk, H]
        q_h = q.mean(dim=3)
        k_h = k.mean(dim=3)
        v_h = v.mean(dim=3)

        # 垂直方向 squeeze，得到 W 维度的轴向特征
        # 形状: [B, Cqk, W]
        q_v = q.mean(dim=2)
        k_v = k.mean(dim=2)
        v_v = v.mean(dim=2)

        # 在 H 轴上做注意力：scores_h [B, H, H]
        scores_h = torch.einsum('bci,bcp->bip', q_h, k_h) * self.scale
        attn_h = F.softmax(scores_h, dim=-1)
        # 输出: [B, Cv, H]
        out_h_tokens = torch.einsum('bip,bcp->bci', attn_h, v_h)
        # broadcast 到 [B, Cv, H, W]
        out_h = out_h_tokens.unsqueeze(-1).expand(-1, -1, -1, W)

        # 在 W 轴上做注意力：scores_v [B, W, W]
        scores_v = torch.einsum('bcj,bcp->bjp', q_v, k_v) * self.scale
        attn_v = F.softmax(scores_v, dim=-1)
        # 输出: [B, Cv, W]
        out_v_tokens = torch.einsum('bjp,bcp->bcj', attn_v, v_v)
        # broadcast 到 [B, Cv, H, W]
        out_v = out_v_tokens.unsqueeze(2).expand(-1, -1, H, -1)

        # 全局语义特征
        global_feat = out_h + out_v  # [B, C, H, W] （Cv == dim）

        # -------- Detail enhancement kernel：局部细节 --------
        enh = self.enh_dwconv(qkv)
        enh = self.enh_bn1(enh)
        enh = F.relu6(enh, inplace=True)
        enh = self.enh_proj(enh)
        enh = self.enh_bn2(enh)
        enh = torch.sigmoid(enh)  # 细节增强权重

        # 结合全局语义与局部细节（乘法融合）
        out = global_feat * enh

        return out


# =========================================================
# Conv-FFN（与 Mix-FFN 类似，但完全在 2D feature map 上操作）
# =========================================================
class ConvFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=True,
        )
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        x = self.pw1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x


# =========================================================
# SeaFormer Block：SEA Attention + Conv-FFN
# （增加 forward_with_maps，用于可视化 CAA / LCFFN）
# =========================================================
class SeaFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # SeaFormer 中更偏向使用 BatchNorm2d/Conv 这种 CNN 友好结构
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SEAAttention(dim)

        self.norm2 = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(dim, hidden_dim, dropout=dropout)

    def forward(self, x, H, W):
        """
        x: [B, N, C], N = H * W
        """
        B, N, C = x.shape
        assert N == H * W, "token 数量必须等于 H*W"

        # tokens -> feature map
        feat = x.transpose(1, 2).reshape(B, C, H, W)

        # 1) SEA Attention
        feat = feat + self.attn(self.norm1(feat))

        # 2) Conv-FFN
        feat = feat + self.mlp(self.norm2(feat))

        # feature map -> tokens
        x = feat.flatten(2).transpose(1, 2)
        return x

    def forward_with_maps(self, x, H, W):
        """
        与 forward 类似，但额外返回注意力输出与 FFN 输出，
        用于可视化 CAA（attention）和 LCFFN（FFN）特征。
        返回:
            x_out: [B, N, C]
            att_out: [B, C, H, W]
            ffn_out: [B, C, H, W]
        """
        B, N, C = x.shape
        assert N == H * W, "token 数量必须等于 H*W"

        feat = x.transpose(1, 2).reshape(B, C, H, W)

        # SEA Attention (CAA)
        norm1_feat = self.norm1(feat)
        att_out = self.attn(norm1_feat)
        feat = feat + att_out

        # Conv-FFN (LCFFN)
        norm2_feat = self.norm2(feat)
        ffn_out = self.mlp(norm2_feat)
        feat = feat + ffn_out

        x_out = feat.flatten(2).transpose(1, 2)
        return x_out, att_out, ffn_out


# =========================================================
# SeaFormer Encoder：4 个 Stage
# =========================================================
class SeaFormerEncoder(nn.Module):
    """
    简化版 SeaFormer 编码器：
    4 个 stage，每个 stage = PatchEmbedding + 多个 SeaFormerBlock
    """
    def __init__(
        self,
        in_channels=3,
        embed_dims=[32, 64, 160, 256],
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        dropout=0.0,
    ):
        super().__init__()

        # Stage 1
        self.patch_embed1 = OverlapPatchEmbedding(
            in_channels, embed_dims[0], patch_size=7, stride=4, padding=3
        )
        self.block1 = nn.ModuleList(
            [
                SeaFormerBlock(embed_dims[0], num_heads[0], dropout=dropout)
                for _ in range(depths[0])
            ]
        )

        # Stage 2
        self.patch_embed2 = OverlapPatchEmbedding(
            embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1
        )
        self.block2 = nn.ModuleList(
            [
                SeaFormerBlock(embed_dims[1], num_heads[1], dropout=dropout)
                for _ in range(depths[1])
            ]
        )

        # Stage 3
        self.patch_embed3 = OverlapPatchEmbedding(
            embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1
        )
        self.block3 = nn.ModuleList(
            [
                SeaFormerBlock(embed_dims[2], num_heads[2], dropout=dropout)
                for _ in range(depths[2])
            ]
        )

        # Stage 4
        self.patch_embed4 = OverlapPatchEmbedding(
            embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1
        )
        self.block4 = nn.ModuleList(
            [
                SeaFormerBlock(embed_dims[3], num_heads[3], dropout=dropout)
                for _ in range(depths[3])
            ]
        )

    def forward(self, x):
        """
        返回四个尺度的 feature map，用于解码头：
        [feat1, feat2, feat3, feat4]
        """
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H1, W1)
        feat1 = x.transpose(1, 2).reshape(B, -1, H1, W1)
        outs.append(feat1)

        # Stage 2
        x, H2, W2 = self.patch_embed2(feat1)
        for blk in self.block2:
            x = blk(x, H2, W2)
        feat2 = x.transpose(1, 2).reshape(B, -1, H2, W2)
        outs.append(feat2)

        # Stage 3
        x, H3, W3 = self.patch_embed3(feat2)
        for blk in self.block3:
            x = blk(x, H3, W3)
        feat3 = x.transpose(1, 2).reshape(B, -1, H3, W3)
        outs.append(feat3)

        # Stage 4
        x, H4, W4 = self.patch_embed4(feat3)
        for blk in self.block4:
            x = blk(x, H4, W4)
        feat4 = x.transpose(1, 2).reshape(B, -1, H4, W4)
        outs.append(feat4)

        return outs

    def forward_with_att(self, x):
        """
        与 forward 类似，但返回：
            outs: 四个尺度特征
            att_maps: 一个 dict，包含每个 stage 最后一层 block 的
                      attention 输出和 FFN 输出，用于可视化。
        """
        B = x.shape[0]
        outs = []
        att_maps = {}

        # Stage 1
        x, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            if i == len(self.block1) - 1:
                x, att, ffn = blk.forward_with_maps(x, H1, W1)
                att_maps["stage1_att"] = att
                att_maps["stage1_ffn"] = ffn
            else:
                x = blk(x, H1, W1)
        feat1 = x.transpose(1, 2).reshape(B, -1, H1, W1)
        outs.append(feat1)

        # Stage 2
        x, H2, W2 = self.patch_embed2(feat1)
        for i, blk in enumerate(self.block2):
            if i == len(self.block2) - 1:
                x, att, ffn = blk.forward_with_maps(x, H2, W2)
                att_maps["stage2_att"] = att
                att_maps["stage2_ffn"] = ffn
            else:
                x = blk(x, H2, W2)
        feat2 = x.transpose(1, 2).reshape(B, -1, H2, W2)
        outs.append(feat2)

        # Stage 3
        x, H3, W3 = self.patch_embed3(feat2)
        for i, blk in enumerate(self.block3):
            if i == len(self.block3) - 1:
                x, att, ffn = blk.forward_with_maps(x, H3, W3)
                att_maps["stage3_att"] = att
                att_maps["stage3_ffn"] = ffn
            else:
                x = blk(x, H3, W3)
        feat3 = x.transpose(1, 2).reshape(B, -1, H3, W3)
        outs.append(feat3)

        # Stage 4
        x, H4, W4 = self.patch_embed4(feat3)
        for i, blk in enumerate(self.block4):
            if i == len(self.block4) - 1:
                x, att, ffn = blk.forward_with_maps(x, H4, W4)
                att_maps["stage4_att"] = att
                att_maps["stage4_ffn"] = ffn
            else:
                x = blk(x, H4, W4)
        feat4 = x.transpose(1, 2).reshape(B, -1, H4, W4)
        outs.append(feat4)

        return outs, att_maps


# =========================================================
# 解码头：类似 SegFormer Head
# =========================================================
class SeaFormerHead(nn.Module):
    """
    解码头：
    - 将 4 个尺度特征映射到统一通道 decoder_dim
    - 上采样到同一尺度拼接
    - 再卷积 + 上采样回输入大小
    """
    def __init__(self, embed_dims=[32, 64, 160, 256], decoder_dim=256, out_channels=1):
        super().__init__()

        self.proj1 = nn.Conv2d(embed_dims[0], decoder_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(embed_dims[1], decoder_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(embed_dims[2], decoder_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(embed_dims[3], decoder_dim, kernel_size=1)

        self.fuse = nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(decoder_dim)
        self.act = nn.ReLU(inplace=True)

        self.pred = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def forward(self, features, input_size):
        feat1, feat2, feat3, feat4 = features
        B, C1, H1, W1 = feat1.shape

        _feat1 = self.proj1(feat1)
        _feat2 = F.interpolate(self.proj2(feat2), size=(H1, W1), mode="bilinear", align_corners=False)
        _feat3 = F.interpolate(self.proj3(feat3), size=(H1, W1), mode="bilinear", align_corners=False)
        _feat4 = F.interpolate(self.proj4(feat4), size=(H1, W1), mode="bilinear", align_corners=False)

        x = torch.cat([_feat1, _feat2, _feat3, _feat4], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.pred(x)  # 当前是 1/4 分辨率左右

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


# =========================================================
# 最终接口：UNET（内部改为 SeaFormer）
# =========================================================
class UNET(nn.Module):
    """
    接口保持不变：
        model = UNET(in_channels=3, out_channels=1, features=[64,128,256,512])

    实际内部实现为 SeaFormer 编码器 + 解码头。
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],  # 仅保留接口，不再真正使用
    ):
        super(UNET, self).__init__()

        # 类似 SeaFormer/SegFormer-B0 的配置
        self.embed_dims = [32, 64, 160, 256]

        self.encoder = SeaFormerEncoder(
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            depths=[2, 2, 2, 2],
            num_heads=[1, 2, 4, 8],
            dropout=0.0,
        )

        self.decode_head = SeaFormerHead(
            embed_dims=self.embed_dims,
            decoder_dim=256,
            out_channels=out_channels,
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        输出: [B, out_channels, H, W]
        """
        input_size = x.shape[2:]   # (H, W)
        features = self.encoder(x)
        out = self.decode_head(features, input_size)
        return out

    def forward_with_att(self, x):
        """
        与 forward 类似，但同时返回注意力特征图，用于可视化：
        返回:
            out: 分割输出 [B, 1, H, W]
            att_maps: dict，包含各 stage 的 CAA / LCFFN 特征
        """
        input_size = x.shape[2:]
        features, att_maps = self.encoder.forward_with_att(x)
        out = self.decode_head(features, input_size)
        return out, att_maps


# =========================================================
# 对网络进行测试
# =========================================================
def test():
    x = torch.randn((3, 1, 160, 160))          # [B, C, H, W]
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print("input:", x.shape)
    print("output:", preds.shape)
    assert preds.shape == x.shape, "输出尺寸必须与输入一致"

    # 测试 forward_with_att
    preds2, att_maps = model.forward_with_att(x)
    print("with_att output:", preds2.shape)
    print("att_maps keys:", list(att_maps.keys()))


if __name__ == "__main__":
    test()
