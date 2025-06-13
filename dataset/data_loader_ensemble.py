import os
import pdb

import torch
import numpy as np
import pickle
import librosa
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from collections import defaultdict
from torch.utils import data
from transformers import AutoProcessor
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_euler_angles, euler_angles_to_matrix
from flame_model.FLAME import FLAMEModel
import torch.nn.functional as F

flame    = FLAMEModel(n_shape=300,n_exp=50)

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, subjects_dict, data_type="train", read_audio=False):
        self.data = data
        self.len  = len(self.data)
        self.data_type  = data_type
        self.read_audio = read_audio

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name      = self.data[index]["name"]
        audio          = self.data[index]["audio"]
        audio_features = self.data[index]["audio_features"]
        blendshapes    = self.data[index]["blendshapes"]
        vertice        = self.data[index]["vertice"]
        template       = self.data[index]["template"]
        subject        = file_name.split("_")[0]

        if self.read_audio:
            return torch.FloatTensor(audio), torch.FloatTensor(audio_features), torch.FloatTensor(vertice), torch.FloatTensor(blendshapes), torch.FloatTensor(template), file_name
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(blendshapes), torch.FloatTensor(template), file_name

    def __len__(self):
        return self.len

def get_vertices_from_blendshapes(expr_tensor, gpose_tensor, jaw_tensor, eyelids_tensor=None):

    target_shape_tensor = torch.zeros(expr_tensor.shape[0], 300).expand(expr_tensor.shape[0], -1)

    I = matrix_to_euler_angles(torch.cat([torch.eye(3)[None]], dim=0),"XYZ")

    eye_r    = I.clone().squeeze()
    eye_l    = I.clone().squeeze()
    eyes     = torch.cat([eye_r,eye_l],dim=0).expand(expr_tensor.shape[0], -1)

    pose = torch.cat([gpose_tensor, jaw_tensor], dim=-1)

    flame_output_only_shape,_ = flame.forward(shape_params=target_shape_tensor, 
                                               expression_params=expr_tensor, 
                                               pose_params=pose, 
                                               eye_pose_params=eyes)

    return flame_output_only_shape.detach()

def lowpass_filter(tensor, kernel_size=10):
    # tensor: [seq_len, 3]
    tensor = tensor.unsqueeze(0).transpose(1,2)  # -> [1, 3, seq_len]
    
    # Create a 1D uniform kernel
    kernel = torch.ones(1, 1, kernel_size, device=tensor.device) / kernel_size
    
    # Apply same kernel to each channel (grouped conv)
    filtered = F.conv1d(tensor, kernel.expand(3, -1, -1), padding=kernel_size//2, groups=3)
    
    return filtered.transpose(1,2).squeeze(0)  # -> back to [seq_len, 3]


def apply_temporal_smoothing(data, window_size=15, sigma=2.0):
    """
    Apply Gaussian temporal smoothing to a sequence of pose data.
    Args:
        data: Tensor or NumPy array of shape [N, D] where N is number of frames and D is dimensions
        window_size: Size of the smoothing window (should be odd)
        sigma: Standard deviation for Gaussian kernel
    Returns:
        Smoothed tensor or array of same shape and type as input
    """
    # Convert to tensor if numpy array
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        data = torch.tensor(data, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    device = data.device
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    # Create Gaussian kernel
    kernel_range = torch.arange(window_size, device=device) - (window_size // 2)
    kernel = torch.exp(-(kernel_range ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # Normalize
    # Apply convolution for each dimension
    smoothed_data = torch.zeros_like(data)
    for i in range(data.shape[1]):
        # Extract the channel data
        channel_data = data[:, i]
        # Manual padding with reflection
        pad_size = window_size // 2
        padded_channel = torch.cat([
            torch.flip(channel_data[:pad_size], [0]),  # Reflect left padding
            channel_data,
            torch.flip(channel_data[-pad_size:], [0])  # Reflect right padding
        ])
        # Apply 1D convolution
        smoothed_channel = torch.zeros_like(channel_data)
        for j in range(len(channel_data)):
            # Apply kernel manually
            window = padded_channel[j:j+window_size]
            smoothed_channel[j] = torch.sum(window * kernel)
        smoothed_data[:, i] = smoothed_channel
    # Convert back to numpy if input was numpy
    if is_numpy:
        smoothed_data = smoothed_data.cpu().numpy()
    return smoothed_data

def read_data(args, test_config=False):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_root, args.wav_path)
    vertices_path = os.path.join(args.data_root, args.vertices_path)
    
    template_file = torch.load(args.template_file, map_location="cpu")
    templates = template_file['flame_model']

    if args.read_audio:  # read_audio==False when training vq to save time
        #processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")  # Wav2Vec
        #processor = AutoProcessor.from_pretrained(args.wav2vec2model_path)  
        processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec2model_path)

     
        #feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        #print("Using {} processor".format("facebook/hubert-large-ls960-ft"))
        print("Using {} processor".format(args.wav2vec2model_path))

    cnt=0

    ####spliting train, val, test
    train_txt = open(os.path.join(args.data_root,"train.txt"), "r")
    test_txt  = open(os.path.join(args.data_root,"test.txt"), "r")
    train_lines, test_lines, train_list, test_list = train_txt.readlines(), test_txt.readlines(), [], []

    for tt in train_lines:
        train_list.append(tt.split("\n")[0])
    for tt in test_lines:
        test_list.append(tt.split("\n")[0])

    counter = 0
    for r, ds, fs in os.walk(audio_path):

        for f in tqdm(fs):
            counter += 1
            ###Activate when testing the model
            if test_config and f not in test_list:
                continue

            if f.endswith("wav"):
                if args.read_audio:
                    wav_path = os.path.join(r, f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values_raw            = speech_array
                    input_values_features       = np.squeeze(processor(speech_array, sampling_rate=16000).input_values) 

                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values_raw if args.read_audio else None
                data[key]["audio_features"] = input_values_features if args.read_audio else None
                subject_id = "_".join(key.split("_")[:-1])
    
                data[key]["name"] = f
                
                vertice_path = os.path.join(vertices_path, f.replace("wav", "npz"))

                temp = templates["v_template"]
                data[key]["template"] = temp.reshape((-1))

            
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    flame_param = np.load(vertice_path, allow_pickle=True)
                    
                    expr   = flame_param["exp"].reshape(-1,50)
                    jaw    = flame_param["pose"][:,3:6].reshape(-1,3)
                    gpose  = flame_param["pose"][:,0:3].reshape(-1,3)
                    gpose  = gpose - gpose.mean(axis=0, keepdims=True)

                    # Compute vertices for supervision in vq training
                    exp_tensor   = torch.Tensor(expr)
                    jaw_tensor   = torch.Tensor(jaw)
                    gpose_tensor = torch.Tensor(gpose)
                    eyelids_tensor = torch.ones((exp_tensor.shape[0], 2))

                    
                    #concat_blendshapes = np.concatenate((exp_tensor.numpy(), jaw_tensor.numpy(), gpose_tensor.numpy()), axis=1)
                    concat_blendshapes = np.concatenate((exp_tensor.numpy(), gpose_tensor.numpy(), jaw_tensor.numpy(), eyelids_tensor.numpy()), axis=1)

                    data[key]["blendshapes"] = concat_blendshapes

                    # Compute vertices for supervision in vq training
                    #blendshapes_derived_vertices = get_vertices_from_blendshapes(exp_tensor,jaw_tensor,gpose_tensor)
                    # Compute vertices for supervision in vq training
                    blendshapes_derived_vertices = get_vertices_from_blendshapes(exp_tensor,gpose_tensor, jaw_tensor)

                    data[key]["vertice"] = blendshapes_derived_vertices.reshape((blendshapes_derived_vertices.shape[0], -1)).numpy()

            if counter > 1000:
                break
                    
    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    # train vq and pred
    train_cnt = 0
    for k, v in data.items():
        k_wav = k.replace("npy", "wav")
        if k_wav in train_list:
            #if train_cnt<int(len(train_list)*0.9):
            if train_cnt < int(counter* 0.8):
                train_data.append(v)
            else:
                valid_data.append(v)
            train_cnt+=1
        elif k_wav in test_list:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args, test_config=False):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args, test_config)

    if not test_config:
        train_data = Dataset(train_data, subjects_dict, "train", args.read_audio)
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers)
        valid_data = Dataset(valid_data, subjects_dict, "val", args.read_audio)
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=args.workers)
    test_data = Dataset(test_data, subjects_dict, "test", args.read_audio)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=args.workers)
    return dataset


if __name__ == "__main__":
    get_dataloaders()