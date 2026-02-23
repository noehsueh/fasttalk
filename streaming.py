#!/usr/bin/env python
"""
Real-time audio-to-face-animation streaming pipeline with Gradio visualization

Architecture:
    Thread 1 (Producer):  AudioSource -> AudioBuffer -> FeatureGenerator -> feature_queue
    Thread 2 (Inference): feature_queue -> BlendshapeModel -> blendshape_queue
    Thread 3 (Render):    blendshape_queue -> Renderer -> image_queue
    Main Thread (Gradio): image_queue -> Web UI
"""

import queue
import threading
import time
import os
import shutil
import subprocess
import tempfile
import yaml
from abc import ABC, abstractmethod
from typing import Optional
from types import SimpleNamespace

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import gradio as gr 
from pytorch3d.transforms import matrix_to_euler_angles

from models                import get_model
from base.baseTrainer      import load_state_dict
from models.utils import enc_dec_mask
from flame_model.FLAME import FLAMEModel
from renderer.renderer import Renderer


# ════════════════════════════════════════════════════════════════
# Configuration / Global Setup
# ════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STREAM_OUT_DIR = "demo/video"
os.makedirs(STREAM_OUT_DIR, exist_ok=True)

# Load FLAME and Renderer globally to avoid reloading per run
def load_flame_and_renderer():
    print(f"[Setup] Loading FLAME and Renderer on {DEVICE}...")
    flame = FLAMEModel(n_shape=300, n_exp=50).to(DEVICE)
    renderer = Renderer(render_full_head=True).to(DEVICE)
    return flame, renderer

FLAME_MODEL, FACE_RENDERER = load_flame_and_renderer() # Initialize globally

# ════════════════════════════════════════════════════════════════
# AudioSource
# ════════════════════════════════════════════════════════════════
class AudioSource(ABC):
    """Abstract base class for audio sources"""

    @abstractmethod
    def start(self):
        """Initialize the audio source"""
        pass

    @abstractmethod
    def get_chunk(self) -> Optional[np.ndarray]:
        """
        Returns 40ms of audio samples, None when done
        """
        pass


class FileAudioSource(AudioSource):
    """Reads audio from a wav file and returns 40ms chunks sequentially"""

    def __init__(self, wav_path: str, chunk_ms: int = 40, pad_last_chunk: bool = False):
        self.wav_path = wav_path
        self.chunk_ms = chunk_ms
        self.pad_last_chunk = pad_last_chunk
        self.audio = None
        self.sr = 16000  # Target sample rate
        self.chunk_size = None
        self.position = 0

    def start(self):
        self.audio, self.sr = librosa.load(self.wav_path, sr=16000)
        self.chunk_size = int(self.sr * self.chunk_ms / 1000)
        self.position = 0
        print(f"[FileAudioSource] Loaded {len(self.audio)} samples ({len(self.audio)/self.sr:.2f}s) at {self.sr}Hz")

    def get_chunk(self) -> Optional[np.ndarray]:
        if self.audio is None:
            raise RuntimeError("AudioSource not started. Call start() first.")

        if self.position >= len(self.audio):
            return None

        end_pos = min(self.position + self.chunk_size, len(self.audio))
        chunk = self.audio[self.position:end_pos]
        self.position = end_pos

        # Pad last chunk if needed
        if len(chunk) < self.chunk_size and self.pad_last_chunk:
            print(f"[FileAudioSource] Last chunk {len(chunk)} samples, padding to {self.chunk_size}")
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

        return chunk


class LiveAudioSource(AudioSource):
    """Receives live audio chunks and serves fixed-size frames."""

    def __init__(self, chunk_ms=40, sr=16000, resample_mode='torch'):
        self.sr = sr
        self.chunk_size = int(sr * chunk_ms / 1000)
        self.queue = queue.Queue()
        self._residual = np.zeros(0, np.float32)
        self._closed = False
        self._resample_mode = resample_mode  # or 'librosa'
        self._lock = threading.Lock()

    def start(self):
        pass
    
    def _resample(self, chunk, orig_sr):
        if self._resample_mode == 'torch':
            return AF.resample(torch.from_numpy(chunk), orig_sr, self.sr).numpy()
        else:
            return librosa.resample(chunk, orig_sr=orig_sr, target_sr=self.sr)

    def push_chunk(self, chunk, sr):
        with self._lock:
            if chunk is None:
                return
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            chunk = np.asarray(chunk, np.float32)
            if sr != self.sr:
                chunk = self._resample(chunk, orig_sr=sr)
            chunk = np.concatenate([self._residual, chunk]) if self._residual.size else chunk
            while len(chunk) >= self.chunk_size:
                frame, chunk = chunk[: self.chunk_size], chunk[self.chunk_size :]
                try:
                    self.queue.put_nowait(frame)
                except queue.Full:
                    self.queue.get_nowait()
                    self.queue.put_nowait(frame)
            self._residual = chunk


    def get_chunk(self) -> Optional[np.ndarray]:
        item = self.queue.get()
        return None if item is None else item

    def stop(self):
        self._closed = True
        self.queue.put(None)

# ════════════════════════════════════════════════════════════════
# AudioBuffer
# ════════════════════════════════════════════════════════════════
class AudioBuffer:
    """Accumulates audio chunks with sliding window behavior"""

    def __init__(self, chunk_ms: int = 40, max_duration_ms: int = 5000, sr: int = 16000):
        self.chunk_ms = chunk_ms
        self.max_duration_ms = max_duration_ms
        self.sr = sr
        self.chunk_size = int(sr * chunk_ms / 1000)
        self.max_samples = int(sr * max_duration_ms / 1000)
        self.buffer = np.array([], dtype=np.float32)

    def add_chunk(self, chunk: np.ndarray):
        self.buffer = np.concatenate([self.buffer, chunk])
        if len(self.buffer) > self.max_samples:
            print(f"[AudioBuffer] Buffer exceeded max duration, trimming...")
            self.buffer = self.buffer[-self.max_samples:]

    def get_audio(self) -> np.ndarray:
        return self.buffer.copy()

    def duration_ms(self) -> int:
        return int(len(self.buffer) * 1000 / self.sr)

    def clear(self):
        self.buffer = np.array([], dtype=np.float32)


# ════════════════════════════════════════════════════════════════
# FeatureGenerator
# ════════════════════════════════════════════════════════════════
class FeatureGenerator:
    """Generates Wav2Vec2 features from audio"""

    def __init__(self, model_name: str = "utter-project/mHuBERT-147", device: str = "cuda"):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(model_name)
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()   
        self.audio_encoder.to(device)
        self.audio_encoder.eval()
        print(f"[FeatureGenerator] Loaded {model_name}")

    @torch.no_grad()
    def generate(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        if len(audio) == 0:
            return None

        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.audio_encoder(**inputs).last_hidden_state
        return features


# ════════════════════════════════════════════════════════════════
# BlendshapeModel
# ════════════════════════════════════════════════════════════════
class BlendshapeModel:
    """Predicts blendshapes from features with autoregressive state"""

    def __init__(self, fasttalk_model: torch.nn.Module, style_t: Optional[torch.Tensor] = None):
        self.model = fasttalk_model
        self.model.eval()
        self.style_t = style_t
        self.device = DEVICE
        
        self.style_vector = None
        if style_t is not None:
             self._compute_style_vector()
        else:
             # Fallback style
             self.style_vector = torch.zeros(1, self.model.args.feature_dim, device=self.device)

        self.past_blendshapes = torch.zeros(1,1,1024).to(self.device)
        self.memory = None
        self.max_blendshape_len = 600 # biased mask size

    def _compute_style_vector(self):
        feat_q_gt, _, encoded = self.model.autoencoder.get_quant(self.style_t, torch.ones_like(self.style_t[...,0], dtype=torch.bool))
        style_feats = self.model.style_proj(feat_q_gt) 
        style_feats = self.model.pos_enc(style_feats) 
        style_feats = self.model.style_frame_encoder(style_feats)    
        style_vec   = style_feats.mean(dim=1)       
        style_vec   = F.normalize(style_vec, dim=-1)   #[B, 1024]
        self.style_vector = style_vec
    
    def _remove_past_context(self):
        # por cada embedding tengo 2 frames de memoria/audio
        # when tgt_mask size exceeds max len, remove past context 100 frames
        frames_to_remove = 100
        memory_frames_to_remove = frames_to_remove * 2
        if self.past_blendshapes.size(1) > self.max_blendshape_len:
            print(f"[BlendshapeModel] Removing past context")
            print(f'Before removing shapes size: {self.past_blendshapes.size()}, memory size: {self.memory.size()}')
            self.past_blendshapes = self.past_blendshapes[:, frames_to_remove:, :].detach()
            self.memory = self.memory[:, memory_frames_to_remove:, :].detach()
            print(f'After removing shapes size: {self.past_blendshapes.size()}, memory size: {self.memory.size()}')

    @torch.no_grad()
    def predict(self, features: torch.Tensor, debug: bool = False) -> torch.Tensor:
        # assert features is [..., 2,...]
        if features.shape[1] != 2:
             print(f"[Warning] Predicted expected 2 frames, got {features.shape[1]}")
             
        hidden_state = self.model.audio_feature_map(features) 

        # add current audio frame to memory
        is_first_frame = self.memory is None
        self.memory = torch.cat(
            [self.memory, hidden_state], dim=1
        ) if self.memory is not None else hidden_state

        self._remove_past_context()
        
        # pos encode blendshapes_input 
        blendshapes_input = self.model.pos_enc(self.past_blendshapes) # [1, T, 1024]
        
        # build mask
        tgt_mask = (
            self.model.biased_mask[:, :blendshapes_input.size(1), :blendshapes_input.size(1)]
            .clone().detach().to(self.device).squeeze(0)
        )
        memory_mask = enc_dec_mask(
            self.device, 
            self.model.dataset,
            blendshapes_input.shape[1], 
            self.memory.shape[1] 
        )
        
        # AR decode
        feat_out = self.model.transformer_decoder(
            tgt = blendshapes_input,
            memory = self.memory,
            tgt_mask = tgt_mask,
            memory_mask = memory_mask,
            style = self.style_vector,
        )

        feat_out = self.model.feat_map(feat_out)

        # decode to blendshapes
        # first frame boostrap
        if is_first_frame:
            bs_out_q = self.model.autoencoder.decode(
                torch.cat([feat_out, feat_out], dim=1)
            )
            bs_out_q = bs_out_q[:, 0].unsqueeze(1)  
        else:
            bs_out_q = self.model.autoencoder.decode(feat_out)
        blendshapes = bs_out_q[:,-1,:]

        # update internal state
        self.memory = self.memory.detach()
        blendshape_emb = self.model.blendshapes_map(blendshapes).unsqueeze(1)  # (1,1,1024)
        self.past_blendshapes = torch.cat(
            [self.past_blendshapes, blendshape_emb], dim=1
        ).detach()
        if debug:
            return blendshapes, feat_out
        return blendshapes

    def reset(self):
        self.past_blendshapes = torch.zeros(1,1,1024).to(self.device)
        self.memory = None


# ════════════════════════════════════════════════════════════════
# Renderer
# ════════════════════════════════════════════════════════════════
class OnlineRenderer:
    """
    Renders blendshapes to output frame using FLAME
    """

    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue
        self.frame_count = 0
        self.device = DEVICE
        self.cam = torch.tensor([5, 0, 0], dtype=torch.float32, device=self.device).expand(1, -1)
        self.eye = matrix_to_euler_angles(torch.eye(3)[None], "XYZ").to(self.device)
        self.shp = torch.zeros(1, 300, device=self.device)

    def _verts_from_bs(self, expr, gpose, jaw, eyelids):
        """Helper to get vertices from blendshape parameters"""
        B   = expr.shape[0]
        shp = self.shp.expand(B, -1)
        eye = self.eye
        eyes = torch.cat([eye.squeeze(), eye.squeeze()], 0).expand(B, -1)
        pose = torch.cat([gpose, jaw], -1)
        v, _ = FLAME_MODEL(shape_params=shp, expression_params=expr, pose_params=pose, eye_pose_params=eyes)
        return v.detach()

    @torch.no_grad()
    def render(self, blendshapes: torch.Tensor):
        """
        Render blendshapes to image and put in queue
        Args:
            blendshapes: Tensor of shape (1, 58) or (58,)
        """
        if blendshapes.ndim == 1:
            blendshapes = blendshapes.unsqueeze(0)
            
        # decompose blendshapes
        # indices from gradio_app.py / stage2.py logic:
        # 0-50: expr, 50-53: gpose, 53-56: jaw, 56-: eyelids
        bs = blendshapes.to(self.device)
        expr    = bs[:, :50]
        gpose   = bs[:, 50:53]
        jaw     = bs[:, 53:56]
        eyelids = bs[:, 56:]

        verts = self._verts_from_bs(expr, gpose, jaw, eyelids)
        cam   = self.cam.expand(expr.size(0), -1)
        
        # Render
        rendered_dict = FACE_RENDERER(verts, cam)
        img_tensor = rendered_dict["rendered_img"][0] # [H, W, 3] usually, or [3, H, W]? Check renderer.
        # Renderer usually returns [B, H, W, 3] or [B, H, W, 4]
        
        # Convert to numpy uint8 for Gradio
        img_np = img_tensor.cpu().numpy()
        # if using my renderer.py, check output format. usually it is normalized to 0-1
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        # Transpose from (C, H, W) to (H, W, C) for Gradio/PIL
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # print(f"[Render] Push frame {self.frame_count} shape: {img_np.shape}") 
        self.output_queue.put(img_np)
        self.frame_count += 1


# ════════════════════════════════════════════════════════════════
# StreamingPipeline
# ════════════════════════════════════════════════════════════════
class StreamingPipeline:
    """Coordinates the 3-thread streaming pipeline"""

    def __init__(
        self,
        audio_source: AudioSource,
        audio_buffer: AudioBuffer,
        feature_generator: FeatureGenerator,
        blendshape_model: BlendshapeModel,
        renderer: OnlineRenderer,
        queue_size: int = 500,
        first_packet_latency_ms: int = 360, 
    ):
        self.audio_source = audio_source
        self.audio_buffer = audio_buffer
        self.feature_generator = feature_generator
        self.blendshape_model = blendshape_model
        self.renderer = renderer
        self.warmup = first_packet_latency_ms

        self.feature_queue = queue.Queue(maxsize=queue_size)
        self.blendshape_queue = queue.Queue(maxsize=queue_size)
        
        self.threads = []
        self.stop_event = threading.Event()

    def _producer_thread(self):
        print("[Producer] Starting...")
        warmup_ms = self.warmup
        is_warmed_up = False

        try:
            self.audio_source.start()
            while not self.stop_event.is_set():
                chunk = self.audio_source.get_chunk()
                if chunk is None:
                    # Flush final frames
                    print("[Producer] Audio exhausted, flushing...")
                    audio = self.audio_buffer.get_audio()
                    all_features = self.feature_generator.generate(audio)
                    if all_features is not None and all_features.shape[1] >= 6:
                        final_features = all_features[:, -6:, :]
                        # send in pairs
                        for i in range(final_features.shape[1] // 2):
                            inc = final_features[:, i*2 : i*2+2, :]
                            self.feature_queue.put(inc)
                        # self.feature_queue.put(final_features)
                    self.feature_queue.put(None)
                    break

                self.audio_buffer.add_chunk(chunk)

                if not is_warmed_up:
                    if self.audio_buffer.duration_ms() >= warmup_ms:
                        audio = self.audio_buffer.get_audio()
                        all_features = self.feature_generator.generate(audio)
                        # Warmup: extract all but last 6 frames
                        if all_features is not None:
                            warmup_features = all_features[:, :-6, :]
                            # send in pairs
                            for i in range(warmup_features.shape[1] // 2):
                                inc = warmup_features[:, i*2 : i*2+2, :]
                                self.feature_queue.put(inc)
                            is_warmed_up = True
                            print(f"[Producer] Warmup done")

                elif self.audio_buffer.duration_ms() >= warmup_ms + self.audio_buffer.chunk_ms:
                    audio = self.audio_buffer.get_audio()
                    all_features = self.feature_generator.generate(audio)
                    if all_features is not None:
                         # Incremental: last 8 to 6
                         inc = all_features[:, -8:-6, :]
                         self.feature_queue.put(inc)

        except Exception as e:
            print(f"[Producer] Error: {e}")
            import traceback
            traceback.print_exc()
            self.feature_queue.put(None)

    def _inference_thread(self):
        print("[Inference] Starting...")
        # try:
        while not self.stop_event.is_set():
            features = self.feature_queue.get()
            if features is None:
                self.blendshape_queue.put(None)
                break
            blendshapes = self.blendshape_model.predict(features)
            self.blendshape_queue.put(blendshapes)
        # except Exception as e:
        #     print(f"[Inference] Error: {e}")
        #     self.blendshape_queue.put(None)

    def _render_thread(self):
        print("[Render] Starting...")
        try:
            while not self.stop_event.is_set():
                blendshapes = self.blendshape_queue.get()
                if blendshapes is None:
                    # signal end to renderer queue maybe?
                    self.renderer.output_queue.put(None)
                    break
                self.renderer.render(blendshapes)
        except Exception as e:
            print(f"[Render] Error: {e}")
            self.renderer.output_queue.put(None)

    def start_non_blocking(self):
        """Start threads but don't join them"""
        self.threads = [
            threading.Thread(target=self._producer_thread, name="Producer"),
            threading.Thread(target=self._inference_thread, name="Inference"),
            threading.Thread(target=self._render_thread, name="Render")
        ]
        for t in self.threads:
            t.start()
            
    def stop(self):
        self.stop_event.set()
        if hasattr(self.audio_source, "stop"):
            try:
                self.audio_source.stop()
            except Exception:
                pass
        for t in self.threads:
            t.join()


# ════════════════════════════════════════════════════════════════
# Main / Gradio
# ════════════════════════════════════════════════════════════════

def flat_yaml(path):
    raw = yaml.safe_load(open(path))
    return SimpleNamespace(**{k: v for sec in raw.values() if isinstance(sec, dict) for k, v in sec.items()})

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
    fps: float,
    output_path: str,
) -> str:
    """
    Combine a directory of numbered PNG frames with an audio track into an MP4.
    """
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
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return output_path


def main_gradio():
    # 1. Load Model (cached globally if possible, or loaded here)
    cfg = flat_yaml("config/joint_data/demo.yaml")
    fasttalk_model = get_model(cfg).to(DEVICE)
    ckpt = torch.load(cfg.model_path, map_location="cpu")
    load_state_dict(fasttalk_model, ckpt["state_dict"], strict=False)
    fasttalk_model.eval()
    # fasttalk_model = torch.compile(fasttalk_model)
    # fasttalk_model.transformer_decoder = torch.compile(fasttalk_model.transformer_decoder)

    # Load style list
    STYLE_DIR = "demo/styles"
    styles = sorted(f[:-4] for f in os.listdir(STYLE_DIR) if f.endswith(".npz"))
    
    from collections import deque

    def init_pipeline(style_name):
        audio_source = LiveAudioSource(chunk_ms=40)
        audio_buffer = AudioBuffer(chunk_ms=40)
        feature_generator = FeatureGenerator(device="cuda" if torch.cuda.is_available() else "cpu")
        # feature_generator.audio_encoder = torch.compile(feature_generator.audio_encoder)
        style_fp = os.path.join(STYLE_DIR, style_name + ".npz")
        style_t = torch.load(style_fp, map_location=DEVICE)

        bs_model = BlendshapeModel(fasttalk_model, style_t)
        #TODO refactor
        feature_generator.audio_encoder = bs_model.model.audio_encoder
        image_queue = queue.Queue()
        renderer = OnlineRenderer(image_queue)

        pipeline = StreamingPipeline(
            audio_source, audio_buffer, feature_generator, bs_model, renderer
        )
        #pipeline.start_non_blocking()

        return {
            "audio_source": audio_source,
            "pipeline": pipeline,
            "image_queue": image_queue,
            "style_name": style_name,
            "frame_buffer":deque(),
            "started": False,
        }

    def on_audio_stream(audio, state):
        if audio is None:
            if state is not None:
                state["pipeline"].stop()
            return gr.update(), None
        # if style_name is None:
        #     return gr.update(), state

        if state['started'] is False:
            state["pipeline"].start_non_blocking()
            state['started'] = True
            state['accumulated_audio'] = []

        # if state is None or state.get("style_name") != style_name:
        #     if state is not None:
        #         state["pipeline"].stop()
        #     state = init_pipeline(style_name)

        sr, samples = audio
        state["audio_source"].push_chunk(samples, sr)
        state['accumulated_audio'].append(samples)
        state['sample_rate'] = sr

        frame = None
        while True:
            try:
                state["frame_buffer"].append(state["image_queue"].get_nowait())
                # frame = state["image_queue"].get_nowait()
                # state['last_frame'] = frame
            except queue.Empty:
                break
        
        MIN_START_FRAMES = 3
        if not state["started"]:
            if len(state["frame_buffer"]) >= MIN_START_FRAMES:
                state["started"] = True
            else:
                # not enough buffered yet: show last frame (or None)
                return state.get("last_frame", None), state
        
        frame = state["frame_buffer"].popleft() if len(state["frame_buffer"]) > 0 else None
        state['last_frame'] = frame
        if frame is None:
            return state.get('last_frame', None), state
        return frame, state

    def apply_style(style_name, state):
        if state is not None:
            state['pipeline'].stop()
        
        new_state = init_pipeline(style_name)
        return new_state, gr.update(visible=True, interactive=True)


    def run_file_stream(audio_path, style_name):
        if not audio_path or style_name is None:
            yield None, None
            return

        audio_source = FileAudioSource(audio_path, chunk_ms=40)
        audio_buffer = AudioBuffer(chunk_ms=40)
        feature_generator = FeatureGenerator(device="cuda" if torch.cuda.is_available() else "cpu")
        fps = 1000.0 / audio_source.chunk_ms

        style_fp = os.path.join(STYLE_DIR, style_name + ".npz")
        style_t = torch.load(style_fp, map_location=DEVICE)

        bs_model = BlendshapeModel(fasttalk_model, style_t)

        image_queue = queue.Queue()
        renderer = OnlineRenderer(image_queue)

        pipeline = StreamingPipeline(
            audio_source, audio_buffer, feature_generator, bs_model, renderer
        )

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_video_path = os.path.join(STREAM_OUT_DIR, f"{base_name}_stream_audio.mp4")
        frames_in_memory = []
        last_frame = None
        frame_dir = None

        try:
            pipeline.start_non_blocking()
            while True:
                try:
                    img = image_queue.get(timeout=1.0)
                    if img is None:
                        break
                    last_frame = img
                    frames_in_memory.append(img)
                    yield img, None
                except queue.Empty:
                    if not any(t.is_alive() for t in pipeline.threads):
                        break
                    continue

            final_video = None
            if frames_in_memory:
                try:
                    frame_dir = tempfile.mkdtemp(prefix="fasttalk_stream_frames_")
                    for idx, frame in enumerate(frames_in_memory):
                        frame_path = os.path.join(frame_dir, f"{idx:06d}.png")
                        save_frame_png(frame, frame_path)
                    final_video = build_video_from_frames(
                        frames_dir=frame_dir,
                        audio_path=audio_path,
                        fps=fps,
                        output_path=out_video_path,
                    )
                except Exception as exc:
                    print(f"[File Stream] Failed to build video: {exc}")
                finally:
                    if frame_dir is not None:
                        shutil.rmtree(frame_dir, ignore_errors=True)

            yield last_frame, final_video
        finally:
            pipeline.stop()
    

    def process_audio_after_recording(state):
        """Return the accumulated audio when recording stops"""
        if state is None or 'accumulated_audio' not in state:
            return None
        
        # Concatenate all chunks
        full_audio = np.concatenate(state['accumulated_audio'])
        sr = state.get('sample_rate', 16000)

        # resample to 16000 if needed
        if sr != 16000:
            # convert to floa
            full_audio = full_audio.astype(np.float32) / 32768.0
            full_audio = librosa.resample(full_audio, orig_sr=sr, target_sr=16000)

        return (16000, full_audio)



    with gr.Blocks(title="fasTTalk Live Streaming") as demo:
        gr.Markdown("### fasTTalk Live - Streaming Pipeline Demo")

        with gr.Tabs():
            with gr.TabItem("Microphone"):
                state = gr.State(None)

                with gr.Row():
                    with gr.Column():
                        style_dd = gr.Dropdown(choices=styles, value=styles[0] if styles else None, label="Style")
                        init_button = gr.Button("Load Fasttalk and style")
                        audio_in = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            label="Microphone",
                            interactive=False, # start disabled until pipeline init
                            visible=False,
                        )
                    with gr.Column():
                        out_image = gr.Image(label="Live Stream", streaming=True)
                        audio_output = gr.Audio(label="Output")

                init_button.click(
                    fn = apply_style,
                    inputs = [style_dd, state],
                    outputs = [state, audio_in],
                )
                if state is not None:
                    # enable interactive
                    audio_in.interactive = True

                audio_in.stream(
                    on_audio_stream,
                    inputs=[audio_in, state],
                    outputs=[out_image, state],
                    stream_every=0.04, # 40ms
                    time_limit=60,
                )
                audio_in.stop_recording(
                    fn=process_audio_after_recording,
                    inputs=[state],
                    outputs=[audio_output],)

            with gr.TabItem("File"):
                with gr.Row():
                    with gr.Column():
                        audio_file = gr.Audio(
                            sources=["upload","microphone"],
                            type="filepath",
                            label="Input Audio",
                        )
                        style_dd_file = gr.Dropdown(choices=styles, value=styles[0] if styles else None, label="Style")
                        btn = gr.Button("Stream File")

                    with gr.Column():
                        out_image_file = gr.Image(label="File Stream", streaming=True)
                        out_video_file = gr.Video(
                            label="Rendered video (frames + audio)",
                            format="mp4",
                            interactive=False,
                            height=240,
                        )

                btn.click(
                    run_file_stream,
                    inputs=[audio_file, style_dd_file],
                    outputs=[out_image_file, out_video_file],
                )

    demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main_gradio()
