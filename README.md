# ⚡ fasTTalk  
### *Speech-to-FLAME Animation at interactive rates*

## Instalation - Bash Script

Run: 

```bash
chmod +x install_env.sh
bash install_env.sh
```

## Run Gradio App

Run: 

```bash
python gradio_app.py
```

## Checkpoints

Model checkpoints can be downloaded from [HERE](). Unzip the files inside the `checkpoints` directory.

## Instalation - Manual

### Step 1: Create and Activate Conda Environment

```bash
conda create --name fasttalk python=3.11
conda activate fasttalk
```

After creating the enviroment, please download and unzip [FLAME](https://drive.google.com/file/d/1dgDWQB9hbGMrQMTVhIv32s3WZmq1CzbZ/view?usp=sharing) inside the root folder. 


### Step 2: Install Dependencies

1. **Install MPI-IS Mesh Library**

    ```bash
    pip install git+https://github.com/MPI-IS/mesh.git
    ```

2. **Install PyTorch with CUDA Support**

    ```bash
    pip3 install torch torchvision torchaudio
    ```

2. **Install ffmepg**

    ```bash
    conda install -c conda-forge ffmpeg
    ```

4. **Install CUDA Toolkit and Ninja**

    ```bash
    conda install -c "nvidia/label/cuda-12.6" cuda-toolkit ninja
    ```

5. **Set Up CUDA Path**

    ```bash
    ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64" 
    conda env config vars set CUDA_HOME=$CONDA_PREFIX
    ```

6. **Reactivate Environment to Apply Changes**

    ```bash
    conda deactivate
    conda activate fasttalk
    ```

### Step 3: Additional Package Installations

1. **Install PyTorch3D**

    ```bash
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"
    ```

2. **Install Other Python Packages**

    ```bash
    pip install tensorboardX einops scipy librosa tqdm
    ```

3. **Install Specific Versions and Additional Packages**

    ```bash
    pip install sympy==1.13.1 
    pip install transformers 
    pip install trimesh pyrender pyopengl pyglet opencv-python pyyaml scikit-image wandb matplotlib
    conda install -c conda-forge ffmpeg
    ```

4. **Install Chumpy and Datasets**

    ```bash
    pip install chumpy datasets
    ```

### Step 4: Configure Wav2Vec2 for `facebook/wav2vec2-large-xlsr-53`

To use `facebook/wav2vec2-large-xlsr-53`, update the following line in `processing_wav2vec2.py`:

1. Navigate to:
    ```
    /path/to/conda_environment/lib/python3.11/site-packages/transformers/models/wav2vec2/processing_wav2vec2.py
    ```
2. Change:
    ```python
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    ```
   to:
    ```python
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h", **kwargs)
    ```

### Step 5: Comment Out Deprecated Imports in Chumpy

1. Open:
    ```
    /path/to/conda_environment/lib/python3.11/site-packages/chumpy/__init__.py
    ```
2. Comment out line 11:
    ```python
    # from numpy import bool, int, float, complex, object, unicode, str, nan, inf
    ```

### Step 6: Other fixes inside Chumpy

1. Open:
    ```
    /path/to/conda_environment/lib/python3.11/site-packages/chumpy/ch.py
    ```
2. Add this lines below ```import inspect```:
    ```python
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    ```

---

Your environment should now be ready for use!
