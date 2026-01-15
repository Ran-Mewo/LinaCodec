import torch
from linacodec.vocoder.vocos import Vocos
from huggingface_hub import snapshot_download
from .model import LinaCodecModel
from .util import load_audio, vocode
from .voice_pack import FlowFormerVoicePack

class LinaCodec:
    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ## download from hf
        if model_path is None:
            model_path = snapshot_download("YatharthS/LinaCodec")

        ## loads linacodec model
        model = LinaCodecModel.from_pretrained(config_path=f"{model_path}/config.yaml", weights_path=f'{model_path}/model.safetensors').eval().to(self.device)

        ## loads distilled wavlm model, 97m params --> 25m + 18m params
        model.load_distilled_wavlm(f"{model_path}/wavlm_encoder.pth", device=self.device)
        model.distilled_layers = [6, 9]

        ## loads vocoder, based of custom vocos and hifigan model with snake
        vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml").to(self.device)
        vocos.load_state_dict(torch.load(f"{model_path}/vocoder/pytorch_model.bin", map_location=self.device))

        self.model = model
        self.vocos = vocos

    @torch.no_grad()
    def encode_features(self, audio_path, *, return_content=True, return_global=True):
        """Return low-level features (tokens and/or embeddings). See `LinaCodecModel.encode()` for details."""
        audio = load_audio(audio_path, sample_rate=self.model.config.sample_rate).to(self.device)
        return self.model.encode(audio, return_content=return_content, return_global=return_global)

    @torch.no_grad()
    def encode(self, audio_path):
        """Encode audio into discrete content tokens at a rate of 12.5 t/s or 25 t/s and 128 dim global embedding, single codebook (content_token_indices, global_embedding). Use `encode_features()` for content embeddings too."""
        f = self.encode_features(audio_path, return_content=True, return_global=True)
        return f.content_token_indices, f.global_embedding

    @torch.no_grad()
    def decode(self, content_tokens, global_embedding):
        """decodes tokens and embedding into 48khz waveform"""
        content_tokens = content_tokens.to(self.device)
        global_embedding = global_embedding.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            ## decode tokens and embedding to mel spectrogram
            mel_spectrogram = self.model.decode(content_token_indices=content_tokens, global_embedding=global_embedding)

        ## decode mel spectrogram into 48khz audio using custom vocos model
        waveform = vocode(self.vocos, mel_spectrogram.unsqueeze(0))
        return waveform
        
    def convert_voice(self, source_file, reference_file):
        """converts voice timbre, will keep content of source file but timbre of reference file"""

        ## get tokens and embedding
        content_tokens, _ = self.encode(source_file)
        _, ref_global_embedding = self.encode(reference_file)

        ## decode to audio
        audio = self.decode(content_tokens, ref_global_embedding)
        return audio

    @torch.no_grad()
    def convert_voice_pack(self, source_file, voice_pack, *, seed=0, temperature=1.0):
        """
        Convert voice using a voicepack.

        voice_pack: path to a saved `FlowFormerVoicePack` or an instance.
        """
        pack = voice_pack if isinstance(voice_pack, FlowFormerVoicePack) else FlowFormerVoicePack.load(
            voice_pack, device=self.device
        )
        audio = load_audio(source_file, sample_rate=self.model.config.sample_rate).to(self.device)
        src = self.model.encode(audio, return_content=True, return_global=False)
        global_condition = pack.sample(src.content_embedding, seed=seed, temperature=temperature).squeeze(0)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            mel = self.model.decode(
                content_embedding=src.content_embedding, global_embedding=global_condition, target_audio_length=audio.size(0)
            )
        return vocode(self.vocos, mel.unsqueeze(0))
