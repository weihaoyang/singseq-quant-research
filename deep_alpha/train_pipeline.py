"""Dated deep-alpha training with daily IC selection and resumable checkpoints."""
from __future__ import annotations
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from filelock import FileLock
from deep_alpha.daily_sampler import DailyBatchSampler
from evaluation.research_contract import daily_rank_ic


class PearsonCorrelationLoss(nn.Module):
    """Pearson correlation within one date; this is not Spearman rank loss."""
    def forward(self, pred, target):
        p = pred.flatten() - pred.mean()
        y = target.flatten() - target.mean()
        return -(p * y).sum() / ((p.square().sum()+1e-8).sqrt() * (y.square().sum()+1e-8).sqrt())


def train_deep_model(model, train_dataset, val_dataset, job_id="deep_job_001",
                     epochs=10, batch_size=256, lr=1e-3, weight_decay=1e-4,
                     device=None, output_dir=None, seed=42, min_stocks=50,
                     resume=False, heartbeat=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("unsafe job_id")
    root = Path(output_dir or Path(__file__).resolve().parent / "checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    # Cross-process lock, including direct callers outside the experiment runner.
    with FileLock(str(Path(__file__).resolve().parent / "training.lock"), timeout=0):
        return _train(model, train_dataset, val_dataset, job_id, epochs, batch_size,
                      lr, weight_decay, device, root, seed, min_stocks, resume, heartbeat)


def _train(model, train_dataset, val_dataset, job_id, epochs, batch_size,
           lr, weight_decay, device, root, seed, min_stocks, resume, heartbeat):
    tr, va = train_dataset.sample_metadata, val_dataset.sample_metadata
    if tr is None or va is None or len(tr) == 0 or len(va) == 0:
        raise ValueError("nonempty dated datasets required")
    if "label_end_date" not in tr or tr.label_end_date.isna().any():
        raise ValueError("training label maturity dates required")
    if tr.trade_date.max() >= va.trade_date.min() or tr.label_end_date.max() >= va.trade_date.min():
        raise ValueError("training/validation overlap or unpurged training labels")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = PearsonCorrelationLoss()
    sampler = DailyBatchSampler(tr.trade_date.to_numpy(), batch_size, seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    best_path = root / f"{job_id}_best.pt"
    last_path = root / f"{job_id}_last.pt"
    history, best, start = [], -float("inf"), 1
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v): state[k] = v.to(device)
        history, best, start = ck["history"], ck["best"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng"])
        if "cuda_rng" in ck and torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck["cuda_rng"])
        sampler.rng.bit_generator.state = ck["sampler_rng"]
    bf16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
    for epoch in range(start, epochs + 1):
        began = time.monotonic()
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda" if bf16 else "cpu", dtype=torch.bfloat16, enabled=bf16):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss): raise ValueError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if heartbeat: heartbeat(epoch=epoch, batches=len(losses))
        if not losses: raise ValueError("no daily training batches")
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in val_loader: preds.extend(model(x.to(device)).flatten().cpu().tolist())
        frame = va.copy()
        frame["score"] = preds
        frame["target_ret"] = [val_dataset.labels[e-1] for _, e in val_dataset.indices_map]
        daily = daily_rank_ic(frame, min_stocks=min_stocks)
        ic = daily.rank_ic.mean()
        if not np.isfinite(ic): raise ValueError("validation has no eligible daily cross-sections")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_ic": float(ic), "valid_days": int(daily.rank_ic.notna().sum()),
                        "elapsed": round(time.monotonic()-began, 3)})
        if ic > best:
            best = float(ic)
            tmp = best_path.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.replace(best_path)
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "history": history,
              "best": best, "epoch": epoch, "rng": torch.get_rng_state(),
              "sampler_rng": sampler.rng.bit_generator.state}
        if torch.cuda.is_available(): ck["cuda_rng"] = torch.cuda.get_rng_state_all()
        tmp = last_path.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(last_path)
        print(f"[{job_id}] epoch={epoch} daily_validation_IC={ic:.6f}", flush=True)
    if not best_path.exists(): raise ValueError("no trained checkpoint")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return {"job_id": job_id, "best_val_ic": best, "best_checkpoint": str(best_path),
            "history": history, "metric": "mean_daily_spearman", "status": "UNVERIFIED"}
