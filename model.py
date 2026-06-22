from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.proj = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DirectionalDepthwiseConv(nn.Module):
    """Depthwise 1D strip convolution sampled along a fixed 2D direction."""

    def __init__(self, channels: int, length: int, degree: int):
        super().__init__()
        if length % 2 == 0:
            raise ValueError("Directional strip length must be odd.")
        self.length = length
        self.degree = degree % 180
        self.weight = nn.Parameter(torch.empty(channels, length))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.weight, mean=0.0, std=(2.0 / length) ** 0.5)

    @staticmethod
    def _offset_for_direction(offset: int, degree: int) -> tuple[int, int]:
        if degree == 0:
            return 0, offset
        if degree == 90:
            return offset, 0
        if degree == 45:
            return offset, offset
        if degree == 135:
            return offset, -offset
        raise ValueError(f"Unsupported direction: {degree}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.length // 2
        _, channels, height, width = x.shape
        padded = F.pad(x, (pad, pad, pad, pad))
        out = x.new_zeros(x.shape)

        for idx, offset in enumerate(range(-pad, pad + 1)):
            dy, dx = self._offset_for_direction(offset, self.degree)
            y0 = pad + dy
            x0 = pad + dx
            shifted = padded[:, :, y0 : y0 + height, x0 : x0 + width]
            out = out + shifted * self.weight[:, idx].view(1, channels, 1, 1)

        return out + self.bias.view(1, channels, 1, 1)


class DirectionGuidedTopologyAggregation(nn.Module):
    """
    DGTA from the manuscript:
    orientation prior + tangent/normal directional strip aggregation + fallback.
    """

    DIRECTIONS = (0, 45, 90, 135)

    def __init__(
        self,
        dim: int,
        reduction: int = 2,
        tangent_length: int = 15,
        normal_length: int = 5,
    ):
        super().__init__()
        reduced_dim = max(dim // reduction, 16)
        self.reduced_dim = reduced_dim
        self.reduce = ConvBNAct(dim, reduced_dim, kernel_size=1, padding=0)
        self.orientation_head = nn.Sequential(
            ConvBNAct(reduced_dim, reduced_dim, kernel_size=3),
            nn.Conv2d(reduced_dim, len(self.DIRECTIONS), kernel_size=1),
        )
        self.confidence_head = nn.Sequential(
            ConvBNAct(reduced_dim, reduced_dim, kernel_size=3),
            nn.Conv2d(reduced_dim, 1, kernel_size=1),
        )

        self.tangent_convs = nn.ModuleList(
            [
                DirectionalDepthwiseConv(reduced_dim, tangent_length, degree)
                for degree in self.DIRECTIONS
            ]
        )
        self.normal_convs = nn.ModuleList(
            [
                DirectionalDepthwiseConv(reduced_dim, normal_length, degree + 90)
                for degree in self.DIRECTIONS
            ]
        )
        self.fallback = ConvBNAct(
            reduced_dim,
            reduced_dim,
            kernel_size=3,
            groups=reduced_dim,
        )
        self.out_proj = ConvBNAct(reduced_dim, dim, kernel_size=1, padding=0, act=False)

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        u = self.reduce(x)
        orientation = torch.softmax(self.orientation_head(u), dim=1)
        confidence = torch.sigmoid(self.confidence_head(u))

        tangent = 0.0
        normal = 0.0
        for idx, (tangent_conv, normal_conv) in enumerate(
            zip(self.tangent_convs, self.normal_convs)
        ):
            weight = orientation[:, idx : idx + 1]
            tangent = tangent + weight * tangent_conv(u)
            normal = normal + weight * normal_conv(u)

        fallback = self.fallback(u)
        fused = confidence * (tangent + normal) + (1.0 - confidence) * fallback
        out = self.out_proj(fused)

        if not return_maps:
            return out, normal

        maps = {
            "dgta_orientation": orientation,
            "dgta_confidence": confidence,
            "dgta_tangent": tangent,
            "dgta_normal": normal,
            "dgta_fallback": fallback,
        }
        return out, normal, maps


class NormalCalibratedLocalGeometryRefinement(nn.Module):
    """NLGR from the manuscript: normal-gated 3x3/5x5 local refinement."""

    def __init__(
        self,
        dim: int,
        normal_channels: int | None,
        expansion: int = 2,
    ):
        super().__init__()
        hidden_dim = dim * expansion
        self.expand = ConvBNAct(dim, hidden_dim, kernel_size=1, padding=0)
        self.local_3 = ConvBNAct(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            groups=hidden_dim,
        )
        self.local_5 = ConvBNAct(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            groups=hidden_dim,
        )
        self.gate_from_normal = (
            nn.Conv2d(normal_channels, hidden_dim, kernel_size=1)
            if normal_channels is not None
            else None
        )
        self.gate_from_input = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.project = ConvBNAct(hidden_dim, dim, kernel_size=1, padding=0, act=False)

    def forward(
        self,
        x: torch.Tensor,
        normal_cue: torch.Tensor | None = None,
        return_maps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        v = self.expand(x)
        g3 = self.local_3(v)
        g5 = self.local_5(v)

        if normal_cue is None or self.gate_from_normal is None:
            gate = torch.sigmoid(self.gate_from_input(x))
        else:
            if normal_cue.shape[2:] != x.shape[2:]:
                normal_cue = F.interpolate(
                    normal_cue,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate = torch.sigmoid(self.gate_from_normal(normal_cue))

        fused = gate * g3 + (1.0 - gate) * g5
        out = self.project(fused)

        if not return_maps:
            return out

        maps = {
            "nlgr_gate": gate,
            "nlgr_local_3": g3,
            "nlgr_local_5": g5,
            "nlgr_fused": fused,
        }
        return out, maps


class CrackLiteBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        use_dgta: bool = True,
        use_nlgr: bool = True,
        tangent_length: int = 15,
        normal_length: int = 5,
    ):
        super().__init__()
        self.use_dgta = use_dgta
        self.use_nlgr = use_nlgr
        self.norm1 = LayerNorm2d(dim) if use_dgta else None
        self.norm2 = LayerNorm2d(dim) if use_nlgr else None
        self.dgta = (
            DirectionGuidedTopologyAggregation(
                dim,
                tangent_length=tangent_length,
                normal_length=normal_length,
            )
            if use_dgta
            else None
        )
        normal_channels = self.dgta.reduced_dim if self.dgta is not None else None
        self.nlgr = (
            NormalCalibratedLocalGeometryRefinement(
                dim,
                normal_channels=normal_channels,
            )
            if use_nlgr
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        maps = {}
        y = x
        normal_cue = None

        if self.dgta is not None and self.norm1 is not None:
            if return_maps:
                dgta_out, normal_cue, dgta_maps = self.dgta(self.norm1(x), True)
                maps.update(dgta_maps)
            else:
                dgta_out, normal_cue = self.dgta(self.norm1(x), False)
            y = x + dgta_out

        z = y
        if self.nlgr is not None and self.norm2 is not None:
            if return_maps:
                nlgr_out, nlgr_maps = self.nlgr(self.norm2(y), normal_cue, True)
                maps.update(nlgr_maps)
            else:
                nlgr_out = self.nlgr(self.norm2(y), normal_cue, False)
            z = y + nlgr_out

        if return_maps:
            return z, maps
        return z


class CrackLiteEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, int, int, int] = (32, 64, 160, 256),
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        use_dgta: bool = True,
        use_nlgr: bool = True,
        tangent_length: int = 15,
        normal_length: int = 5,
    ):
        super().__init__()
        self.patch_embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(in_channels, embed_dims[0], 7, 4, 3),
                OverlapPatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 1),
                OverlapPatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 1),
                OverlapPatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 1),
            ]
        )
        self.stages = nn.ModuleList()
        for dim, depth in zip(embed_dims, depths):
            self.stages.append(
                nn.ModuleList(
                    [
                        CrackLiteBlock(
                            dim,
                            use_dgta=use_dgta,
                            use_nlgr=use_nlgr,
                            tangent_length=tangent_length,
                            normal_length=normal_length,
                        )
                        for _ in range(depth)
                    ]
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict]:
        features = []
        maps = {}

        for stage_idx, (patch_embed, blocks) in enumerate(
            zip(self.patch_embeds, self.stages),
            start=1,
        ):
            x = patch_embed(x)
            for block_idx, block in enumerate(blocks, start=1):
                should_collect = return_maps and block_idx == len(blocks)
                if should_collect:
                    x, block_maps = block(x, return_maps=True)
                    for name, value in block_maps.items():
                        maps[f"stage{stage_idx}_{name}"] = value
                else:
                    x = block(x, return_maps=False)
            features.append(x)

        if return_maps:
            return features, maps
        return features


class CrackLiteDecoder(nn.Module):
    def __init__(
        self,
        embed_dims: tuple[int, int, int, int] = (32, 64, 160, 256),
        decoder_dim: int = 128,
        out_channels: int = 1,
        use_aux: bool = True,
    ):
        super().__init__()
        self.use_aux = use_aux
        self.projections = nn.ModuleList(
            [nn.Conv2d(dim, decoder_dim, kernel_size=1) for dim in embed_dims]
        )
        self.fuse = nn.Sequential(
            ConvBNAct(decoder_dim * 4, decoder_dim, kernel_size=1, padding=0),
            ConvBNAct(decoder_dim, decoder_dim, kernel_size=3),
        )
        self.main_head = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)
        if use_aux:
            self.centerline_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
            self.boundary_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],
        input_size: tuple[int, int],
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        f1, f2, f3, f4 = features
        high_size = f1.shape[2:]

        projected = [
            self.projections[0](f1),
            F.interpolate(
                self.projections[1](f2),
                size=high_size,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.projections[2](f3),
                size=high_size,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                self.projections[3](f4),
                size=high_size,
                mode="bilinear",
                align_corners=False,
            ),
        ]
        decoder_feature = self.fuse(torch.cat(projected, dim=1))
        logits = self.main_head(decoder_feature)
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        aux = {}
        if return_aux and self.use_aux:
            aux["centerline"] = F.interpolate(
                self.centerline_head(decoder_feature),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
            aux["boundary"] = F.interpolate(
                self.boundary_head(decoder_feature),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits, aux, decoder_feature


class CrackLite(nn.Module):
    """
    Lightweight crack segmentation network from CL-Manuscript0622.

    The auxiliary centerline and boundary heads are returned only when
    return_aux=True, so inference uses the deployment-time main branch.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        embed_dims: tuple[int, int, int, int] = (32, 64, 160, 256),
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        decoder_dim: int = 128,
        use_dgta: bool = True,
        use_nlgr: bool = True,
        use_aux: bool = True,
        tangent_length: int = 15,
        normal_length: int = 5,
    ):
        super().__init__()
        self.encoder = CrackLiteEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            use_dgta=use_dgta,
            use_nlgr=use_nlgr,
            tangent_length=tangent_length,
            normal_length=normal_length,
        )
        self.decode_head = CrackLiteDecoder(
            embed_dims=embed_dims,
            decoder_dim=decoder_dim,
            out_channels=out_channels,
            use_aux=use_aux,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor | dict:
        input_size = x.shape[2:]
        features = self.encoder(x)
        logits, aux, decoder_feature = self.decode_head(
            features,
            input_size,
            return_aux=return_aux,
        )

        if not return_aux and not return_features:
            return logits

        outputs = {"out": logits}
        if return_aux:
            outputs["aux"] = aux
        if return_features:
            outputs["features"] = features
            outputs["decoder_feature"] = decoder_feature
        return outputs

    def forward_with_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        input_size = x.shape[2:]
        features, maps = self.encoder(x, return_maps=True)
        logits, _, _ = self.decode_head(features, input_size, return_aux=False)
        return logits, maps


class UNET(CrackLite):
    """Backward-compatible alias for older scripts in this repository."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] | None = None,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels)


def test():
    x = torch.randn(2, 3, 128, 128)
    model = CrackLite(in_channels=3, out_channels=1)
    y = model(x)
    outputs = model(x, return_aux=True)
    y_maps, maps = model.forward_with_maps(x)
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("aux:", {k: tuple(v.shape) for k, v in outputs["aux"].items()})
    print("maps:", list(maps.keys())[:4], "...")
    assert y.shape == (2, 1, 128, 128)
    assert y_maps.shape == y.shape


if __name__ == "__main__":
    test()
