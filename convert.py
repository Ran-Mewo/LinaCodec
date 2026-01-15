import argparse
from pathlib import Path

import soundfile as sf
import torch

from linacodec.codec import LinaCodec


def main():
    ap = argparse.ArgumentParser(description="Voice conversion using LinaCodec and a trained voice pack.")
    ap.add_argument("--source", required=True, help="Source wav (content)")
    ap.add_argument("--voice_pack", required=True, help="Path to trained voice pack (.pt)")
    ap.add_argument("--output", required=True, help="Output wav path")
    ap.add_argument("--model_path", default=None, help="Local LinaCodec model folder; if omitted, downloads from HF")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed (set to change style variation)")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more variation)")
    ap.add_argument("--sr", type=int, default=48000, help="Output sample rate")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lina = LinaCodec(model_path=args.model_path, device=device)
    audio = lina.convert_voice_pack(args.source, args.voice_pack, seed=args.seed, temperature=args.temperature)
    y = audio.squeeze(0).detach().cpu().to(torch.float32).numpy()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, args.sr)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

