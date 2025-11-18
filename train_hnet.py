import os
import glob
import json
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from hnet.models.config_hnet import HNetConfig
from hnet.models.mixer_seq import HNetLM


# ---------- Dataset over your prepped chunks ----------

class FineWebChunkDataset(Dataset):
    """
    Loads preprocessed FineWeb-Edu .pt chunks from data_fineweb_bytes/.
    Each file should contain {"x": LongTensor[L], "y": LongTensor[L]}.
    """
    def __init__(self, root: str = "data_fineweb_bytes"):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(root, "chunk_*.pt")))
        if not self.files:
            raise ValueError(f"No chunk_*.pt files found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d = torch.load(self.files[idx])
        x = d["x"].long()
        y = d["y"].long()
        return x, y


def make_dataloader(
    root: str = "data_fineweb_bytes",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = FineWebChunkDataset(root)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------- Config + model helpers ----------

def load_config(config_path: str) -> HNetConfig:
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    # Adjust if HNetConfig has a different factory method in the repo
    return HNetConfig(**cfg_dict)


def build_model(config_path: str, device: torch.device) -> HNetLM:
    cfg = load_config(config_path)
    model = HNetLM(cfg)
    return model.to(device)


# ---------- Training loop ----------

def train(
    config_path: str,
    data_root: str = "data_fineweb_bytes",
    batch_size: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_steps: int = 2000,
    log_every: int = 20,
    save_every: int = 500,
    out_dir: str = "checkpoints",
    device: str = "cuda",
):
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loader = make_dataloader(
        root=data_root,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = build_model(config_path, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    step = 0
    while step < max_steps:
        for x, y in loader:
            step += 1
            x = x.to(device)  # [B, L]
            y = y.to(device)  # [B, L]

            optimizer.zero_grad()
            logits = model(x)  # expected [B, L, vocab_size]

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % log_every == 0:
                print(f"step {step} | loss {loss.item():.4f}")

            if step % save_every == 0:
                ckpt_path = os.path.join(out_dir, f"step_{step:06d}.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": step,
                        "config_path": config_path,
                    },
                    ckpt_path,
                )
                print("Saved checkpoint to", ckpt_path)

            if step >= max_steps:
                break

    ckpt_path = os.path.join(out_dir, "final.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config_path": config_path,
        },
        ckpt_path,
    )
    print("Training complete, final checkpoint at", ckpt_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to H-Net JSON config, e.g. configs/hnet_1stage_L.json",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data_fineweb_bytes",
        help="Directory containing chunk_*.pt",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    train(
        config_path=args.config_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        out_dir=args.out_dir,
        device=args.device,
    )

