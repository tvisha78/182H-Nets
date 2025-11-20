import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch import optim

from hnet_tiny import HNetTinyLM  # assumes hnet_tiny.py is in the same folder


def load_text_bytes(path: str) -> torch.Tensor:
    """
    Load a text file and return a 1D tensor of byte IDs in [0, 255].
    """
    text = Path(path).read_text(encoding="utf-8")
    data = text.encode("utf-8")
    return torch.tensor(list(data), dtype=torch.long)


def make_batches(data: torch.Tensor, seq_len: int, batch_size: int, device):
    """
    Turn a long 1D sequence into fixed-length (B, seq_len) chunks.

    We just chop from the front and discard any remainder that does not fit a full batch.
    """
    num_tokens = data.size(0)
    num_full = num_tokens // (seq_len * batch_size)
    usable_tokens = num_full * seq_len * batch_size
    data = data[:usable_tokens]
    data = data.view(batch_size, -1)  # (B, T_total)

    num_chunks = data.size(1) // seq_len
    chunks = []
    for i in range(num_chunks):
        start = i * seq_len
        end = start + seq_len
        chunks.append(data[:, start:end])
    return [c.to(device) for c in chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.txt", help="Path to plain text file")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha-ratio", type=float, default=0.03, help="Weight on ratio loss")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default="hnet_tiny.pt")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    data = load_text_bytes(args.data)
    print(f"Loaded {data.numel()} bytes from {args.data}")

    batches = make_batches(data, seq_len=args.seq_len, batch_size=args.batch_size, device=device)
    if not batches:
        raise ValueError("Not enough data to form a single batch. Use a larger text file or smaller seq-len/batch-size.")
    print(f"Created {len(batches)} chunks of shape ({args.batch_size}, {args.seq_len})")

    model = HNetTinyLM(
        vocab_size=256,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(batches)
        for batch in batches:
            input_ids = batch  # (B, L)
            logits, aux = model(input_ids)

            # Next-token prediction
            logits_flat = logits[:, :-1, :].contiguous().view(-1, model.vocab_size)
            targets = input_ids[:, 1:].contiguous().view(-1)

            lm_loss = F.cross_entropy(logits_flat, targets)
            ratio_loss = aux["ratio_loss"]
            loss = lm_loss + args.alpha_ratio * ratio_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % 50 == 0:
                print(
                    f"step {global_step} | epoch {epoch} | "
                    f"lm_loss {lm_loss.item():.4f} | ratio_loss {ratio_loss.item():.4f} | "
                    f"total {loss.item():.4f}"
                )
            global_step += 1

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
        },
        args.save,
    )
    print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
