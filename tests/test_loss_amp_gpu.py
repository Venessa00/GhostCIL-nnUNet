#!/usr/bin/env python3
"""Train a tiny network for 200 steps with nnU-Net's mixed-precision setup (torch.autocast
fp16 + GradScaler) using the SAHL loss and check that the loss never becomes NaN/inf.

    python tests/test_loss_amp_gpu.py

Falls back to an fp16 check on CPU when no GPU is available."""
import sys
import torch
import torch.nn as nn

from ghostcil_nnunet.losses import SAHLLoss


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.LeakyReLU(),
                        nn.Conv2d(16, 2, 3, padding=1)).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None
    loss_fn = SAHLLoss(batch_dice=True)
    bad = 0
    for step in range(200):
        x = torch.randn(2, 1, 128, 128, device=dev)
        y = torch.zeros(2, 1, 128, 128, dtype=torch.long, device=dev)
        y[:, :, 40:90, 30:100] = 1
        if step % 3 == 0:                       # occasionally an all-background slice (flat regions)
            y.zero_()
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
            scaler.step(opt); scaler.update()
        else:
            out = net(x).half()
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        if not torch.isfinite(loss):
            bad += 1
    if bad:
        print(f"FAILED: {bad}/200 steps produced a NaN/inf loss")
        sys.exit(1)
    print(f"OK: 200 mixed-precision steps on {dev} with finite loss (last loss={loss.item():.4f})")


if __name__ == "__main__":
    main()
