import os
import torch
from datasets import load_dataset

SEQ_LEN = 2048          # can increase later (4096/8192)
N_CHUNKS = 1000         # just to test pipeline first
OUT_DIR = "data_fineweb_bytes"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-100BT",
        split="train",
        streaming=True,
    )

    buffer = bytearray()
    chunk_idx = 0

    for row in ds:
        text = row.get("text") or ""
        if not text:
            continue

        buffer.extend(text.encode("utf-8"))

        while len(buffer) >= SEQ_LEN + 1 and chunk_idx < N_CHUNKS:
            # take SEQ_LEN+1 bytes so we can do next-byte prediction
            chunk = buffer[: SEQ_LEN + 1]
            del buffer[: SEQ_LEN + 1]

            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)

            out_path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:06d}.pt")
            torch.save({"x": x, "y": y}, out_path)
            print("wrote", out_path)

            chunk_idx += 1

        if chunk_idx >= N_CHUNKS:
            break

    print("Done, wrote", chunk_idx, "chunks to", OUT_DIR)

if __name__ == "__main__":
    main()
    # Hack: avoid weird C++ finalizers that crash on interpreter shutdown
    import os as _os
    _os._exit(0)

