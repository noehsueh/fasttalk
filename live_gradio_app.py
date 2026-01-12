#!/usr/bin/env python
"""
fasTTalk Live - Simulated streaming pipeline

This stage hardcodes a wav file and slices it into non-overlapping 40 ms hops.
Each hop is a forward pass to the model (or mock) and we render that frame.
No sliding context is used in this simplified loop.
"""

import os
import subprocess
import time
import shutil
import tempfile
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Tuple
import warnings

import gradio as gr
import librosa
import numpy as np
import torch
import yaml
from pytorch3d.transforms import matrix_to_euler_angles
from transformers import Wav2Vec2FeatureExtractor

from base.baseTrainer import load_state_dict
from flame_model.FLAME import FLAMEModel
from models import get_model
from renderer.renderer import Renderer

# Audio / model constants
SAMPLE_RATE = 16000
HOP_LENGTH = 640  # 40 ms at 16 kHz
CHUNK_MS = HOP_LENGTH / SAMPLE_RATE * 1000
TARGET_FPS = 25.0  # Match hop (40 ms)

# Paths
CONFIG_PATH = "config/joint_data/demo.yaml"
DEFAULT_SIM_WAV = "/home/lab0/SpeechAvatars/DATASETS/TalkVid/TalkVid-bench/Tracked/output_language/wav/videovideo3121_4S86yoe5RDo-scene224_scene2.wav"
DEFAULT_STYLE = None  # e.g., "neutral" to load demo/styles/neutral.npz
LIVE_OUT_DIR = "demo/live_output"

# Smoothing (disabled for now)
EMA_ALPHA = 0.65

# Torch defaults for inference
torch.set_grad_enabled(False)

# Suppress noisy loader warnings
warnings.filterwarnings("ignore", message="No mtl file provided")
warnings.filterwarnings("ignore", message="Using torch.cross without specifying the dim arg is deprecated.")

# Rendering setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flame = FLAMEModel(n_shape=300, n_exp=50).to(DEVICE)
renderer = Renderer(render_full_head=True).to(DEVICE)
BASE_CAM = torch.tensor([5, 0, 0], dtype=torch.float32, device=DEVICE)
os.makedirs(LIVE_OUT_DIR, exist_ok=True)


def flat_yaml(path: str) -> SimpleNamespace:
    raw = yaml.safe_load(open(path))
    return SimpleNamespace(
        **{k: v for sec in raw.values() if isinstance(sec, dict) for k, v in sec.items()}
    )


def safe_audio_window(window: np.ndarray, min_samples: int) -> np.ndarray:
    """Pad audio to a minimum length for model stability."""
    if len(window) >= min_samples:
        return window
    return np.pad(window, (0, min_samples - len(window)))


def split_blendshape_vector(vec: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "expr": vec[:50],
        "gpose": vec[50:53],
        "jaw": vec[53:56],
        "eyelids": vec[56:],
    }


def verts_from_bs(expr, gpose, jaw, eyelids):
    """Map blendshapes to FLAME vertices."""
    B = expr.shape[0]
    shp = torch.zeros(B, 300, device=DEVICE)
    eye = matrix_to_euler_angles(torch.eye(3, device=DEVICE)[None], "XYZ")
    eyes = torch.cat([eye.squeeze(), eye.squeeze()], 0).expand(B, -1)
    pose = torch.cat([gpose, jaw], -1)
    v, _ = flame(
        shape_params=shp,
        expression_params=expr,
        pose_params=pose,
        eye_pose_params=eyes,
    )
    return v.detach()


def render_blendshapes(bs: Dict[str, np.ndarray]) -> np.ndarray:
    """Render a single frame given blendshape dict."""
    expr = torch.tensor(bs["expr"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    gpose = torch.tensor(bs["gpose"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    jaw = torch.tensor(bs["jaw"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    eyelids = torch.tensor(bs["eyelids"], dtype=torch.float32, device=DEVICE).unsqueeze(0)

    verts = verts_from_bs(expr, gpose, jaw, eyelids)
    cam = BASE_CAM.expand(expr.size(0), -1)
    img = renderer(verts, cam)["rendered_img"][0]
    np_img = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return np_img


class BlendshapeEMA:
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.state: Optional[Dict[str, np.ndarray]] = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, current: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.state is None:
            self.state = current
            return current

        smoothed = {}
        for k, v in current.items():
            smoothed[k] = self.alpha * v + (1 - self.alpha) * self.state[k]
        self.state = smoothed
        return smoothed


class FastTalkModelWrapper:
    """
    Encapsulates model + feature extractor init and per-window inference.
    Always runs the real model. Keeps an audio context window so streaming
    calls can reuse autoregressive history instead of isolated hops.
    """

    def __init__(self, cfg_path: str = CONFIG_PATH, device: Optional[str] = None):
        self.cfg_path = cfg_path
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg: Optional[SimpleNamespace] = None
        self.model = None
        self.feature_extractor = None
        self.loaded = False
        self.context_window = 50  # frames
        self.hop_length = HOP_LENGTH
        self.context_samples = self.context_window * self.hop_length
        self.style_cache: Dict[str, torch.Tensor] = {}

    def load(self) -> None:
        if self.loaded:
            return

        self.cfg = flat_yaml(self.cfg_path)
        self.context_window = getattr(self.cfg, "interactive_window", self.context_window)
        self.hop_length = getattr(self.cfg, "hop_length", self.hop_length)
        self.context_samples = self.context_window * self.hop_length
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.wav2vec2model_path)

        self.model = get_model(self.cfg).to(self.device)
        ckpt = torch.load(self.cfg.model_path, map_location="cpu")
        load_state_dict(self.model, ckpt["state_dict"], strict=False)
        self.model.eval()
        self.loaded = True

    def load_style(self, style_name: Optional[str]) -> Optional[torch.Tensor]:
        if style_name is None:
            return None
        style_fp = os.path.join("demo/styles", style_name + ".npz")
        if not os.path.exists(style_fp):
            return None
        if style_name in self.style_cache:
            return self.style_cache[style_name]
        style = torch.load(style_fp, map_location=self.device)
        self.style_cache[style_name] = style
        return style

    def predict_last_frame(
        self,
        audio_buffer: np.ndarray,
        target_style: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        if not self.loaded:
            self.load()

        # Only keep the most recent context window worth of samples
        if len(audio_buffer) > self.context_samples:
            audio_buffer = audio_buffer[-self.context_samples :]

        # Ensure minimum length so frame_num >= 1 inside model
        min_samples = self.hop_length * 2
        padded_window = safe_audio_window(audio_buffer, min_samples)

        inputs = self.feature_extractor(
            padded_window,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        feats = inputs.input_values.to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                pred = self.model.predict_no_quantizer(feats, target_style=target_style).squeeze(0)
        if pred.shape[0] == 0:
            raise RuntimeError("Model returned zero frames for the provided audio window.")
        last_frame = pred[-1].detach().cpu().numpy()
        return split_blendshape_vector(last_frame)


class OfflineAudioStreamer:
    """
    Simulates online audio by chunking a pre-saved wav into hop-sized frames (no overlap).
    """

    def __init__(
        self,
        wav_path: str,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
    ):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        audio, sr = librosa.load(wav_path, sr=sample_rate)
        self.audio = audio.astype(np.float32)
        self.total_frames = int(np.ceil(len(self.audio) / hop_length))

    def frames(self) -> Iterable[Dict]:
        for frame_idx in range(self.total_frames):
            start = frame_idx * self.hop_length
            end = start + self.hop_length
            window = self.audio[start:end]
            if len(window) < self.hop_length:
                # Pad tail of audio so final chunk is still hop-sized
                window = np.pad(window, (0, self.hop_length - len(window)))

            yield {
                "frame_idx": frame_idx,
                "timestamp": end / self.sample_rate,
                "audio_window": window,
            }


def copy_audio_local(wav_path: str) -> str:
    """
    Copy external wav into a temp file so Gradio can serve it.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="fasttalk_live_")
    os.close(fd)
    shutil.copyfile(wav_path, tmp_path)
    return tmp_path


def save_frame_png(img: np.ndarray, path: str) -> None:
    """
    Save an RGB uint8 numpy image to disk.
    """
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, img)
        return
    except Exception:
        pass

    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save frame {path}: {exc}")


def build_video_from_frames(
    frames_dir: str,
    audio_path: str,
    fps: float = TARGET_FPS,
    output_path: Optional[str] = None,
    max_res: int = 512,
) -> str:
    """
    Combine a directory of numbered PNG frames with an audio track into an MP4.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="fasttalk_live_")
        os.close(fd)

    pattern = os.path.join(frames_dir, "%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "28",
        "-vf",
        f"scale='min({max_res},iw)':-2",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to mux video: {exc}")

    return output_path


def simulate_stream_from_wav(
    wav_path: str,
    model_bundle: FastTalkModelWrapper,
    style_name: Optional[str] = DEFAULT_STYLE,
    smoother: Optional[BlendshapeEMA] = None,
) -> Iterable[Dict]:
    style_tensor = model_bundle.load_style(style_name)
    streamer = OfflineAudioStreamer(wav_path)
    audio_buffer = np.zeros(0, dtype=np.float32)

    for step in streamer.frames():
        t0 = time.time()
        # Accumulate audio into a rolling buffer to preserve context
        audio_buffer = np.concatenate([audio_buffer, step["audio_window"]]).astype(np.float32)
        bs = model_bundle.predict_last_frame(audio_buffer, target_style=style_tensor)
        latency_ms = (time.time() - t0) * 1000

        if smoother is not None:
            bs = smoother(bs)

        yield {
            "frame_idx": step["frame_idx"],
            "timestamp": step["timestamp"],
            "latency_ms": latency_ms,
            "blendshapes": bs,
            "audio_window": step["audio_window"],
        }


# Gradio state
model_bundle = FastTalkModelWrapper()
try:
    model_bundle.load()
except Exception as exc:
    warnings.warn(f"Failed to load pretrained model on startup: {exc}")
# EMA smoothing is disabled for comparison with offline inference.
smoother = None


def run_simulated_stream(
    wav_path: str,
) -> Iterable[Tuple[Optional[np.ndarray], str, Optional[str]]]:
    """
    Render each predicted frame as soon as a chunk is processed, save all frames,
    then mux them with the input audio into an MP4 at the end.
    """
    global model_bundle, smoother

    try:
        local_wav = copy_audio_local(wav_path)
    except FileNotFoundError:
        yield None, f"File not found: {wav_path}", None
        return
    except Exception as e:
        yield None, f"Failed to prepare audio: {e}", None
        return

    # Reset smoother (disabled) and load real model
    model_bundle.load()

    frame_dir = tempfile.mkdtemp(prefix="fasttalk_frames_")
    last_frame: Optional[np.ndarray] = None
    frame_count = 0
    out_video_path = os.path.join(LIVE_OUT_DIR, f"live_{int(time.time())}.mp4")

    try:
        for step in simulate_stream_from_wav(local_wav, model_bundle, smoother=smoother):
            frame_img = render_blendshapes(step["blendshapes"])
            frame_path = os.path.join(frame_dir, f"{step['frame_idx']:06d}.png")
            save_frame_png(frame_img, frame_path)
            last_frame = frame_img
            frame_count += 1

            status = (
                f"Frame {step['frame_idx']} | t={step['timestamp']:.3f}s "
                f"| latency={step['latency_ms']:.1f}ms "
                f"| saving to {frame_path}"
            )
            # Yield current frame and status; video is not ready yet
            yield frame_img, status, None

        if frame_count == 0:
            yield None, "No frames were produced.", None
            return

        try:
            video_path = build_video_from_frames(
                frames_dir=frame_dir,
                audio_path=local_wav,
                fps=TARGET_FPS,
                output_path=out_video_path,
            )
            status = (
                f"Rendered {frame_count} frames at {TARGET_FPS:.1f} fps | "
                f"saved video: {video_path}"
            )
            yield last_frame, status, video_path
        except Exception as exc:
            yield last_frame, f"Rendering failed: {exc}", None
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
        try:
            os.remove(local_wav)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#                               GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="fasTTalk Live - Simulated Streaming") as demo:
    gr.Markdown("### fasTTalk Live — Simulated online inference from wav")
    gr.Markdown(
        "Uses a hardcoded wav, slices non-overlapping 40 ms hops, runs the model per hop, "
        "and renders the newest frame each step. Toggle model loading when ready."
    )

    wav_path_box = gr.Textbox(
        label="Simulated wav path",
        value=DEFAULT_SIM_WAV,
        placeholder="demo/audio/your_file.wav",
    )

    status_label = gr.Label(label="Status", value="Idle")
    with gr.Row():
        frame_view = gr.Image(label="Rendered frame", type="numpy")
        video_player = gr.Video(
            label="Rendered video (frames + audio)",
            format="mp4",
            interactive=False,
            height=200,
            scale=1,
        )

    run_btn = gr.Button("Run simulated stream")

    run_btn.click(
        run_simulated_stream,
        inputs=[wav_path_box],
        outputs=[frame_view, status_label, video_player],
    )

if __name__ == "__main__":
    print("Starting fasTTalk Live simulated streaming...")
    print(f"Sample Rate: {SAMPLE_RATE} Hz | Hop: {HOP_LENGTH} samples ({CHUNK_MS:.1f} ms)")
    print(f"Default wav: {DEFAULT_SIM_WAV}")
    print("Launch the Gradio UI to begin.\n")
    demo.launch()
