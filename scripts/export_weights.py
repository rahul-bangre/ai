"""
Export Qwen/Qwen2.5-0.5B-Instruct into the QWN1 float32 file the C++ loader reads.

Usage:
    python scripts/export_weights.py --out weights/qwen2.5-0.5b-instruct.f32.bin
"""

import argparse
import struct
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAGIC = b"QWN1"
VERSION = 1
DTYPE_F32 = 0


def u32(out, value: int) -> None:
    out.write(struct.pack("<I", value))


def f32(out, value: float) -> None:
    out.write(struct.pack("<f", value))


def tensor(out, name: str, value: torch.Tensor) -> None:
    array = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
    u32(out, len(name.encode("utf-8")))
    out.write(name.encode("utf-8"))
    u32(out, array.ndim)
    for dim in array.shape:
        u32(out, int(dim))
    u32(out, DTYPE_F32)
    out.write(array.numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--out",
        default="weights/qwen2.5-0.5b-instruct.f32.bin",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    state = model.state_dict()

    skip = {"lm_head.weight"} if config.tie_word_embeddings else set()
    tensors = [
        (name, value)
        for name, value in state.items()
        if name not in skip and value.is_floating_point()
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as out:
        out.write(MAGIC)
        u32(out, VERSION)
        u32(out, config.vocab_size)
        u32(out, config.hidden_size)
        u32(out, config.intermediate_size)
        u32(out, config.num_hidden_layers)
        u32(out, config.num_attention_heads)
        u32(out, config.num_key_value_heads)
        u32(out, config.max_position_embeddings)
        u32(out, config.bos_token_id)
        u32(out, config.eos_token_id)
        f32(out, float(config.rms_norm_eps))
        f32(out, float(config.rope_theta))
        u32(out, int(bool(config.tie_word_embeddings)))
        u32(out, len(tensors))
        for name, value in tensors:
            tensor(out, name, value)

    print(f"wrote {len(tensors)} tensors to {out_path}")
    print(f"eos_token_id={config.eos_token_id}")


if __name__ == "__main__":
    main()
