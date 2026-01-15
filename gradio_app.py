#!/usr/bin/env python
"""
fasTTalk demo - audio-driven facial animation (clean 2x2 layout)
"""

import os, subprocess, yaml, gradio as gr
import numpy as np, torch, librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from types import SimpleNamespace
from transformers import Wav2Vec2FeatureExtractor
from pytorch3d.transforms import matrix_to_euler_angles

# flame / renderer / model imports unchanged …
from flame_model.FLAME import FLAMEModel
from renderer.renderer     import Renderer
from models                import get_model
from base.baseTrainer      import load_state_dict

# ────────── Runtime setup ──────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flame    = FLAMEModel(n_shape=300, n_exp=50).to(device)
renderer = Renderer(render_full_head=True).to(device)

ENC_DIR, AUDIO_DIR, OUT_DIR = "demo/output", "demo/audio", "demo/video"
for d in [ENC_DIR, AUDIO_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ────────── Load joint config / model ──────────
def flat_yaml(path):
    raw = yaml.safe_load(open(path))
    return SimpleNamespace(**{k: v for sec in raw.values() if isinstance(sec, dict) for k, v in sec.items()})

cfg   = flat_yaml("config/joint_data/demo.yaml")
model = get_model(cfg).to(device)
ckpt  = torch.load(cfg.model_path, map_location="cpu")
load_state_dict(model, ckpt["state_dict"], strict=False)
model.eval()

# ────────── Helper functions ──────────
def verts_from_bs(expr, gpose, jaw, eyelids):
    B   = expr.shape[0]
    shp = torch.zeros(B, 300, device=device)
    eye = matrix_to_euler_angles(torch.eye(3)[None], "XYZ").to(device)
    eyes = torch.cat([eye.squeeze(), eye.squeeze()], 0).expand(B, -1)
    pose = torch.cat([gpose, jaw], -1)
    v, _ = flame(shape_params=shp, expression_params=expr, pose_params=pose, eye_pose_params=eyes)
    return v.detach()

def render_mp4(npz_path, wav_path, fps=25):
    base = os.path.splitext(os.path.basename(npz_path))[0]
    d    = np.load(npz_path)
    expr, gpose, jaw, eyelids = (torch.tensor(d[k], dtype=torch.float32, device=device)
                                 for k in ("expr", "gpose", "jaw", "eyelids"))
    verts = verts_from_bs(expr, gpose, jaw, eyelids)
    cam   = torch.tensor([5, 0, 0], dtype=torch.float32, device=device).expand(expr.size(0), -1)
    frames = renderer(verts, cam)["rendered_img"]

    fig, ax = plt.subplots(figsize=(5,5))
    def upd(i):
        ax.clear(); ax.imshow((frames[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)); ax.axis("off")
    ani = animation.FuncAnimation(fig, upd, frames=frames.size(0), interval=100)

    tmp = os.path.join(OUT_DIR, f"{base}.mp4")
    ani.save(tmp, writer="ffmpeg", fps=fps); plt.close(fig)

    final = os.path.join(OUT_DIR, f"{base}_audio.mp4")
    subprocess.run(["ffmpeg","-y","-i",tmp,"-i",wav_path,"-c:v","copy","-c:a","aac",final],
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return final

def pipeline(audio_fp, style_name):
    # 1. wav → features
    wav, _ = librosa.load(audio_fp, sr=16000)
    fe     = Wav2Vec2FeatureExtractor.from_pretrained(cfg.wav2vec2model_path)
    feats  = torch.FloatTensor(fe(wav, sampling_rate=16000).input_values).to(device)

    # 2. load style
    style_fp = os.path.join("demo/styles", style_name + ".npz")
    style_t  = torch.load(style_fp, map_location=device)

    # 3. predict blendshapes
    with torch.no_grad():
        bs = model.predict_no_quantizer(feats, target_style=style_t).squeeze(0)

    # 4. save .npz
    name = os.path.splitext(os.path.basename(audio_fp))[0]
    npz_out = os.path.join(ENC_DIR, name + ".npz")
    np.savez(npz_out,
             expr    = bs[:, :50].cpu().numpy(),
             gpose   = bs[:, 50:53].cpu().numpy(),
             jaw     = bs[:, 53:56].cpu().numpy(),
             eyelids = bs[:, 56:].cpu().numpy())

    # 5. keep audio + render
    wav_copy = os.path.join(AUDIO_DIR, name + ".wav")
    if not os.path.exists(wav_copy):
        os.rename(audio_fp, wav_copy)
    vid = render_mp4(npz_out, wav_copy)
    return vid, npz_out

# ────────── Style assets ──────────
STYLE_DIR, PREV_DIR = "demo/styles", "demo/style_previews"
styles = sorted(f[:-4] for f in os.listdir(STYLE_DIR) if f.endswith(".npz"))

def preview_path(style):                        # try .mp4 then images
    for ext in (".mp4",".png",".jpg",".jpeg"):
        p = os.path.join(PREV_DIR, style + ext)
        if os.path.exists(p): return p
    return None

# ══════════════ Gradio UI ══════════════
with gr.Blocks(title="fasTTalk demo") as demo:
    gr.Markdown("### fasTTalk - Audio-driven FLAME animation")

    # Row 1 ─────────────────────────────
    with gr.Row():
        # left-top: audio + generate
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="Upload audio (wav / mp3)")
            gen_btn  = gr.Button("Generate")
        # right-top: video preview
        video_out = gr.Video(label="Generated preview", height=500)

    # Row 2 ─────────────────────────────
    with gr.Row():
        # left-bottom: style selector + style preview side-by-side
        with gr.Column():
            with gr.Row():
                style_radio = gr.Radio(styles, value=styles[0], label="Style")
                style_prev  = gr.Video(label="Style preview", height=200)
        # right-bottom: .npz download
        npz_file = gr.File(label="Download blendshapes (.npz)")

    # Callbacksnpz_file
    style_radio.change(lambda s: gr.update(value=preview_path(s)),
                       inputs=style_radio, outputs=style_prev)

    def run(audio_fp, style_choice):
        vid, npz = pipeline(audio_fp, style_choice)
        return vid, npz

    gen_btn.click(run,
                  inputs=[audio_in, style_radio],
                  outputs=[video_out, npz_file])

if __name__ == "__main__":
    demo.launch()
