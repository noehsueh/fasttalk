#!/usr/bin/env python
# ssh -i ~/.ssh/ssh-simli-vast/ssh-simli-vast -p 17149 root@80.188.223.202 -L 7860:localhost:7860

import os, sys, subprocess
import gradio as gr
import numpy as np
import torch, librosa, pickle, yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from transformers import Wav2Vec2Processor
from pytorch3d.transforms import matrix_to_euler_angles
from flame.flame import FlameHead
from renderer.renderer import Renderer
from types import SimpleNamespace
from models import get_model
from base.baseTrainer import load_state_dict

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flame  = FlameHead(shape_params=300, expr_params=50).to(device)
renderer = Renderer(render_full_head=True).to(device)

ENC_DIR, AUDIO_DIR, OUT_DIR = "demo/output", "demo/audio", "demo/video"
for d in [ENC_DIR, AUDIO_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────
def load_and_flatten_yaml(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    flat = {k: v for sect in cfg.values() if isinstance(sect, dict) for k, v in sect.items()}
    return SimpleNamespace(**flat)

cfg = load_and_flatten_yaml("config/multi/demo.yaml")
model = get_model(cfg).to(device)
if not os.path.isfile(cfg.model_path):
    raise RuntimeError(f"No checkpoint found at {cfg.model_path}")
ckpt = torch.load(cfg.model_path, map_location="cpu")
load_state_dict(model, ckpt["state_dict"], strict=False)
model.eval()

# ────────────────────────────────────────────────
# Render helpers
# ────────────────────────────────────────────────
def get_vertices_from_blendshapes(expr, jaw, neck=None):
    # Load the encoded file
    expr_tensor =  expr.to(device)
    jaw_tensor  =  jaw.to(device) #torch.zeros(expr_tensor.shape[0],3).to(device)

    target_shape_tensor = torch.zeros(expr_tensor.shape[0], 300).expand(expr_tensor.shape[0], -1).to(device)

    I = matrix_to_euler_angles(torch.cat([torch.eye(3)[None]], dim=0),"XYZ").to(device)

    eye_r    = I.clone().to(device).squeeze()
    eye_l    = I.clone().to(device).squeeze()
    eyes     = torch.cat([eye_r,eye_l],dim=0).expand(expr_tensor.shape[0], -1).to(device)

    translation = torch.zeros(expr_tensor.shape[0], 3).to(device)

    if neck==None:
        neck = I.clone().expand(expr_tensor.shape[0], -1).to(device)
    
    rotation = I.clone().expand(expr_tensor.shape[0], -1).to(device)

    # Compute Flame
    flame_output_only_shape   = flame.forward(target_shape_tensor, expr_tensor, rotation, neck, jaw_tensor, eyes, translation, return_landmarks=False)

    return flame_output_only_shape.detach()


def _update_plot(frame_inx, renderer_output_blendshapes, axes):
    # Select the frames to plot
    frame = renderer_output_blendshapes['rendered_img'][frame_inx].detach().cpu().numpy().transpose(1, 2, 0)

    # Update the second subplot
    axes.clear()
    axes.imshow((frame * 255).astype(np.uint8))
    axes.axis('off')
    axes.set_title(f'Frame Stage 1 (Blendshape) {frame_inx + 1}')

# Function to create and save the video
def create_and_save_video(npz_path, wav_path):
    base = os.path.splitext(os.path.basename(npz_path))[0]
    print(base)
    
    blendshapes_data_encoded_expr = np.load(npz_path)['expr'].reshape(-1, 50)
    blendshapes_data_encoded_jaw  = np.load(npz_path)['jaw'].reshape(-1, 3)
    blendshapes_data_encoded_neck  = np.load(npz_path)['neck'].reshape(-1, 3)

    blendshapes_data_encoded_expr = torch.tensor(blendshapes_data_encoded_expr, dtype=torch.float32).to(device)
    blendshapes_data_encoded_jaw  = torch.tensor(blendshapes_data_encoded_jaw, dtype=torch.float32).to(device)
    blendshapes_data_encoded_neck = torch.tensor(blendshapes_data_encoded_neck, dtype=torch.float32).to(device)
    
    # Compute vertices from blendshapes
    blendshapes_derived_vertices = get_vertices_from_blendshapes(blendshapes_data_encoded_expr,blendshapes_data_encoded_jaw, blendshapes_data_encoded_neck)
    
    # Fixed camera
    cam_original = torch.tensor([10,0,0], dtype=torch.float32).expand(blendshapes_derived_vertices.shape[0], -1).to(device)

    # Render the frames
    renderer_output_blendshapes  = renderer.forward(blendshapes_derived_vertices, cam_original)

    N = renderer_output_blendshapes['rendered_img'].shape[0] # Number of frames

    fig, ax = plt.subplots(figsize=(5,5))
    ani = animation.FuncAnimation(fig, _update_plot, frames=N, fargs=(renderer_output_blendshapes, ax), interval=40)
    vid_tmp = os.path.join(OUT_DIR, f"{base}.mp4")
    ani.save(vid_tmp, writer="ffmpeg", fps=25)
    plt.close(fig)

    out_vid = os.path.join(OUT_DIR, f"{base}_with_audio.mp4")
    if os.path.exists(wav_path):
        cmd = ["ffmpeg", "-y", "-i", vid_tmp, "-i", wav_path, "-c:v", "copy", "-c:a", "aac", out_vid]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return out_vid
    return vid_tmp

# ────────────────────────────────────────────────
# Audio pipeline
# ────────────────────────────────────────────────
def process_audio(wav_path):
    # (1) features
    speech, _ = librosa.load(wav_path, sr=16000)
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    feat = torch.FloatTensor(proc(speech, sampling_rate=16000).input_values).to(device)

    # (2) template
    tpl_file = os.path.join("assets/FLAME2023", cfg.template_file)
    tpl = torch.FloatTensor(pickle.load(open(tpl_file, "rb"), encoding="latin1")["v_template"].ravel()).to(device).unsqueeze(0)

    # (3) predicciones
    with torch.no_grad():
        _, expr = model.predict(feat, tpl)
        expr, jaw, neck = torch.split(expr, [50, 3, 3], dim=-1)

    # (4) guardado .npz
    out_name = os.path.splitext(os.path.basename(wav_path))[0]
    npz_path = os.path.join(ENC_DIR, f"{out_name}.npz")
    np.savez(npz_path, expr=expr.cpu().numpy(), jaw=jaw.cpu().numpy(), neck=neck.cpu().numpy())

    # (5) render video
    wav_copy = os.path.join(AUDIO_DIR, f"{out_name}.wav")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    if not os.path.exists(wav_copy):
        os.rename(wav_path, wav_copy)          # conservar el audio original
    vid_path = create_and_save_video(npz_path, wav_copy)
    return npz_path, vid_path

# ────────────────────────────────────────────────
# Gradio app
# ────────────────────────────────────────────────
def gradio_interface(audio_file):
    npz_path, vid_path = process_audio(audio_file)
    return vid_path, vid_path, npz_path          # <─ now returns three items

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Audio(type="filepath", label="Upload WAV/MP3"),
    outputs=[
        gr.Video(label="Preview (mp4)"),         # <─ new inline preview
        gr.File(label="Download .mp4"),           # <─ keep download option
        gr.File(label="Download blendshape .npz")
    ],
    title="fasTTalk demo",
    description="Upload an audio file and get a rendered facial animation"
)

if __name__ == "__main__":
    demo.launch()