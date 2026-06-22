from __future__ import annotations

import time

import torch

from config import IMAGE_HEIGHT, IMAGE_WIDTH
from model import CrackLite, DirectionalDepthwiseConv


def count_directional_depthwise(module, inputs, output):
    x = inputs[0]
    batch, channels, height, width = x.shape
    macs = batch * channels * height * width * module.length
    module.total_ops += torch.DoubleTensor([macs])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrackLite(in_channels=3, out_channels=1).to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, device=device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params / 1e6:.3f} M")

    try:
        from thop import clever_format, profile

        custom_ops = {DirectionalDepthwiseConv: count_directional_depthwise}
        macs, thop_params = profile(
            model,
            inputs=(dummy_input,),
            custom_ops=custom_ops,
            verbose=False,
        )
        flops = macs * 2
        macs_str, params_str, flops_str = clever_format(
            [macs, thop_params, flops],
            "%.3f",
        )
        print(f"THOP Params: {params_str}")
        print(f"MACs       : {macs_str}")
        print(f"FLOPs      : {flops_str}")
    except ImportError:
        print("Install thop to report MACs/FLOPs: pip install thop")

    warmup = 10
    repeats = 50
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeats):
            _ = model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - start

    print(f"FPS: {repeats / total_time:.2f}")


if __name__ == "__main__":
    main()
