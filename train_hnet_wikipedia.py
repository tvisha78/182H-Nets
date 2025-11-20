import argparse
from pathlib import Path
import random
import glob

import torch
import torch.nn.functional as F
from torch import optim
import pyarrow.parquet as pq

from hnet_tiny import HNetTinyLM


def load_wikipedia_sample(
    data_dir: str,
    language: str = "en",
    max_files: int = 1,
    max_articles: int = 1000,
    max_bytes: int = None,
):
    """
    Load a sample from the Wikipedia dataset.
    
    Args:
        data_dir: Path to wikipedia-20231101-all directory
        language: Language code (e.g., "en", "de", "fr")
        max_files: Maximum number of parquet files to load
        max_articles: Maximum number of articles to load
        max_bytes: Maximum number of bytes to load (None = no limit)
    
    Returns:
        torch.Tensor: 1D tensor of byte IDs in [0, 255]
    """
    data_dir = Path(data_dir)
    lang_dir = data_dir / f"20231101.{language}"
    
    if not lang_dir.exists():
        raise ValueError(f"Language directory not found: {lang_dir}")
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(str(lang_dir / "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {lang_dir}")
    
    # Limit number of files
    parquet_files = parquet_files[:max_files]
    print(f"Loading from {len(parquet_files)} parquet file(s)...")
    
    all_texts = []
    total_articles = 0
    
    for parquet_file in parquet_files:
        print(f"  Reading {Path(parquet_file).name}...")
        table = pq.read_table(parquet_file)
        
        # Extract text column
        if 'text' not in table.column_names:
            print(f"    Warning: 'text' column not found in {parquet_file}, skipping")
            continue
        
        texts = table['text'].to_pylist()
        
        for text in texts:
            if text is None or not isinstance(text, str):
                continue
            
            # Add newline separator between articles
            all_texts.append(text + "\n\n")
            total_articles += 1
            
            if total_articles >= max_articles:
                break
        
        if total_articles >= max_articles:
            break
    
    # Concatenate all texts
    full_text = "".join(all_texts)
    print(f"Loaded {total_articles} articles, {len(full_text)} characters")
    
    # Convert to bytes
    data = full_text.encode("utf-8")
    
    # Limit bytes if specified
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
        print(f"Limited to {max_bytes} bytes")
    
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
    parser = argparse.ArgumentParser(description="Train HNet on Wikipedia dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="wikipedia-20231101-all",
        help="Path to wikipedia-20231101-all directory",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (e.g., 'en', 'de', 'fr')",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Maximum number of parquet files to load",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=1000,
        help="Maximum number of articles to load",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Maximum number of bytes to load (None = no limit)",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--alpha-ratio", type=float, default=0.03, help="Weight on ratio loss"
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default="hnet_tiny_wikipedia.pt")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load Wikipedia data
    print(f"\nLoading Wikipedia sample...")
    print(f"  Language: {args.language}")
    print(f"  Max files: {args.max_files}")
    print(f"  Max articles: {args.max_articles}")
    if args.max_bytes:
        print(f"  Max bytes: {args.max_bytes}")
    
    data = load_wikipedia_sample(
        data_dir=args.data_dir,
        language=args.language,
        max_files=args.max_files,
        max_articles=args.max_articles,
        max_bytes=args.max_bytes,
    )
    print(f"\nLoaded {data.numel()} bytes total")

    batches = make_batches(
        data, seq_len=args.seq_len, batch_size=args.batch_size, device=device
    )
    if not batches:
        raise ValueError(
            "Not enough data to form a single batch. Use more articles or smaller seq-len/batch-size."
        )
    print(f"Created {len(batches)} batches of shape ({args.batch_size}, {args.seq_len})")

    model = HNetTinyLM(
        vocab_size=256,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    print(f"\nStarting training...")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}\n")

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
    print(f"\nSaved model to {args.save}")


if __name__ == "__main__":
    main()

