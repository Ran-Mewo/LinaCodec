import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from linacodec.model import LinaCodecModel
from linacodec.module.audio_feature import MelSpectrogramFeature
from linacodec.util import load_audio
from linacodec.voice_pack import FlowFormerVoicePack


def load_linacodec_model(model_path, device):
    model_path = model_path or snapshot_download("YatharthS/LinaCodec")
    model = (
        LinaCodecModel.from_pretrained(
            config_path=f"{model_path}/config.yaml", weights_path=f"{model_path}/model.safetensors"
        )
        .eval()
        .to(device)
    )
    model.load_distilled_wavlm(f"{model_path}/wavlm_encoder.pth", device=device)
    model.distilled_layers = [6, 9]
    return model


def _crop_1d(x: torch.Tensor, length: int, *, gen: torch.Generator) -> torch.Tensor:
    if (n := x.numel()) < length:
        return torch.nn.functional.pad(x, (0, length - n))
    if n == length:
        return x
    start = int(torch.randint(0, n - length + 1, (1,), generator=gen).item())
    return x[start : start + length]


def main():
    ap = argparse.ArgumentParser(description="Train a LinaCodec Voice Pack (Flow-former normalizing flow model).")
    ap.add_argument("--audio_dir", required=True, help="Directory of target voice .wav files")
    ap.add_argument("--output", required=True, help="Where to save the voice pack (.pt)")
    ap.add_argument("--model_path", default=None, help="Local LinaCodec model folder; if omitted, downloads from HF")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of wav files used (0 = no limit)")
    ap.add_argument("--segment_seconds", type=float, default=2.0, help="Random crop length in seconds (pads if shorter)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond_len", type=int, default=128, help="Resampled conditioning length (tokens)")
    ap.add_argument("--dim", type=int, default=256, help="Flow-former hidden dim")
    ap.add_argument("--tr_layers", type=int, default=2)
    ap.add_argument("--tr_heads", type=int, default=8)
    ap.add_argument("--coupling_layers", type=int, default=8)
    ap.add_argument("--clamp", type=float, default=2.0)
    ap.add_argument("--delta_clamp", type=float, default=1.0)
    ap.add_argument("--flow_weight", type=float, default=1.0)
    ap.add_argument("--mel_weight", type=float, default=1.0)
    ap.add_argument("--delta_l2_weight", type=float, default=1e-4)
    ap.add_argument("--delta_tv_weight", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_linacodec_model(args.model_path, device)
    for p in model.parameters():
        p.requires_grad = False

    mel_extractor = MelSpectrogramFeature(
        sample_rate=model.config.sample_rate,
        n_fft=model.config.n_fft,
        hop_length=model.config.hop_length,
        n_mels=model.config.n_mels,
        padding=model.config.padding,
    ).to(device).eval()

    files = sorted([p for p in Path(args.audio_dir).rglob("*.wav") if p.is_file()])
    if args.limit:
        files = files[: int(args.limit)]
    if not files:
        raise SystemExit(f"No .wav files found under: {args.audio_dir}")

    pack = FlowFormerVoicePack(
        cond_dim=model.local_quantizer.output_dim,
        emb_dim=model.global_encoder.output_dim,
        cond_len=args.cond_len,
        dim=args.dim,
        tr_layers=args.tr_layers,
        tr_heads=args.tr_heads,
        coupling_layers=args.coupling_layers,
        clamp=args.clamp,
        delta_clamp=args.delta_clamp,
    ).to(device)
    opt = torch.optim.AdamW(pack.parameters(), lr=args.lr)

    seg_len = int(float(args.segment_seconds) * float(model.config.sample_rate))
    gen = torch.Generator().manual_seed(int(args.seed))

    for epoch in range(1, args.epochs + 1):
        pack.train()
        loss_sum = flow_sum = mel_sum = 0.0
        n = 0
        for pth in tqdm(files, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            wav = _crop_1d(load_audio(str(pth), sample_rate=model.config.sample_rate), seg_len, gen=gen).to(device)
            feat = model.encode(wav, return_content=True, return_global=True)

            flow_loss = pack.loss(feat.global_embedding.unsqueeze(0), feat.content_embedding)
            cond_seq = pack.condition(feat.content_embedding, feat.global_embedding)
            mel_pred = model.decode(
                content_embedding=feat.content_embedding,
                global_embedding=cond_seq,
                target_audio_length=wav.size(0),
            )
            mel_tgt = mel_extractor(wav.unsqueeze(0)).squeeze(0)
            if (t := min(mel_pred.size(-1), mel_tgt.size(-1))) != mel_pred.size(-1):
                mel_pred = mel_pred[..., :t]
            if t != mel_tgt.size(-1):
                mel_tgt = mel_tgt[..., :t]
            mel_loss = torch.nn.functional.l1_loss(mel_pred, mel_tgt)

            delta = cond_seq - feat.global_embedding.view(1, 1, -1)
            delta_l2 = delta.pow(2).mean()
            delta_tv = (delta[:, 1:] - delta[:, :-1]).abs().mean()
            loss = (
                args.flow_weight * flow_loss
                + args.mel_weight * mel_loss
                + args.delta_l2_weight * delta_l2
                + args.delta_tv_weight * delta_tv
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            flow_sum += flow_loss.item()
            mel_sum += mel_loss.item()
            n += 1
        n = max(1, n)
        print(f"epoch={epoch} loss={loss_sum / n:.6f} flow={flow_sum / n:.6f} mel={mel_sum / n:.6f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pack.eval().cpu().save(args.output)
    print(f"Saved voice pack -> {args.output}")


if __name__ == "__main__":
    main()

