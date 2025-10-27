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
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import savgol_filter

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
        blendshapes    = self.data[index]["blendshapes"]

        if self.read_audio:
            audio = self.data[index]["audio"]
            return torch.FloatTensor(blendshapes), torch.FloatTensor(audio)
        else:
            return torch.FloatTensor(blendshapes)

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

    #counter = 0
    frames_count = 0
    for r, ds, fs in os.walk(audio_path):

        for f in tqdm(fs):
            #counter += 1
            # Activate when testing the model
            if test_config and f not in test_list:
                continue

            if f.endswith("wav"):
                key          = f.replace("wav", "npy")
                subject_id   = "_".join(key.split("_")[:-1])
                blendshapes_path = os.path.join(vertices_path, f.replace("wav", "npz"))
            
                if not os.path.exists(blendshapes_path):
                    continue
                else:
                    # Load blendshapes (FLAME parameters)
                    flame_param = np.load(blendshapes_path, allow_pickle=True)

                    # Discard sequences with more than 600 frames (too large for training)
                    #if 'pose' in flame_param and (flame_param["exp"].shape[0] > 350 or flame_param["exp"].shape[0] < 8):
                    #    continue
                    #elif 'pose_params' in flame_param and (flame_param["expression_params"].shape[0] > 350 or flame_param["expression_params"].shape[0] #< 8):
                    #    continue
                    #elif 'gpose' in flame_param and (flame_param["exp"].shape[0] > 350 or flame_param["exp"].shape[0] < 8):
                    #    continue
                    #else:

                    try:
                        expr   = flame_param['exp'].reshape(-1, 50)
                        jaw    = flame_param['jaw'].reshape(-1,3)
                        gpose  = flame_param['pose'].reshape(-1, 3)
                        gpose  = gpose - gpose.mean(axis=0, keepdims=True)

                        # Apply Savitzky-Golay filter along the time axis for gpose (removes tracker's flickering) (axis=0)
                        gpose = savgol_filter(gpose, window_length=7, polyorder=2, axis=0)
                        
                        # Compute vertices for supervision in vq training
                        exp_tensor    = torch.Tensor(expr)
                        jaw_tensor    = torch.Tensor(jaw) 
                        gpose_tensor  = torch.Tensor(gpose)
                        eyelids_tensor = torch.ones((exp_tensor.shape[0], 2)) # Not tracked in all datasets, so we use a placeholder

                        s = torch.empty(1).uniform_(3.0, 4.0) # Headpose exageration factor
                        gpose_tensor *= s
                        
                        concat_blendshapes = np.concatenate((exp_tensor.numpy(), gpose_tensor.numpy(), jaw_tensor.numpy(), eyelids_tensor.numpy()), axis=1)

                        data[key]["blendshapes"] = concat_blendshapes

                        frames_count += concat_blendshapes.shape[0]

                        # Load audio if required
                        if args.read_audio:
                            wav_path = os.path.join(r, f)
                            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                            # Fix to match T video frames (known ahead of time)
                            T = concat_blendshapes.shape[0] # Number of frames in the blendshapes
                            expected_len = T * 640
                            speech_array = librosa.util.fix_length(speech_array, size=expected_len)

                            input_audio_features = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                            data[key]["audio"]   = input_audio_features
                    except Exception as e:
                        print("Error loading data for {}. Skipping.".format(blendshapes_path))
                        continue

            #if counter > 1000:
            #    break
                   
    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"]   = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"]  = [i for i in args.test_subjects.split(" ")]

    # train vq and pred
    train_cnt = 0
    for k, v in data.items():
        k_wav = k.replace("npy", "wav")
        if k_wav in train_list:
            if train_cnt<int(len(train_list)*0.9):
            #if train_cnt < int(counter* 0.7):
                train_data.append(v)
            else:
                valid_data.append(v)
            train_cnt+=1
        elif k_wav in test_list:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    print('Total hours of data: {:.2f}'.format(frames_count / 25 / 60 / 60))
    return train_data, valid_data, test_data, subjects_dict


def collate_fn(batch):
    """
    batch: list of tensors [seq_len_i, 58]
    Returns:
        padded_blendshapes: [B, T_max, 58]
        mask: [B, T_max] — 1 for real tokens, 0 for padding
    """
    # Sort batch by sequence length (optional, helps some models)
    batch = sorted(batch, key=lambda x: x.shape[0], reverse=True)

    # Extract sequence lengths
    lengths = [x.shape[0] for x in batch]

    # Pad sequences to max length
    padded_blendshapes = pad_sequence(batch, batch_first=True, padding_value=0.0)  # [B, T_max, 58]

    # Create mask: 1 where valid, 0 where padding
    mask = torch.zeros(padded_blendshapes.shape[:2], dtype=torch.bool)  # [B, T_max]
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    return padded_blendshapes, mask


def collate_fn_audio(batch):
    """
    batch: list of tuples (blendshapes [T, 58], audio [1, 640*T]) if read_audio=True,
           or list of blendshapes [T, 58] if read_audio=False.

    Returns:
        padded_blendshapes: [B, T_max, 58]
        blendshape_mask:    [B, T_max]
        padded_audio: [B, 1, max_audio_len] (only if audio present)
        audio_mask:   [B, max_audio_len] (only if audio present)
    """
    if isinstance(batch[0], tuple):  # contains (blendshapes, audio)
        blendshapes, audios = zip(*batch)

        # Sort by blendshape length
        sorted_items = sorted(zip(blendshapes, audios), key=lambda x: x[0].shape[0], reverse=True)
        blendshapes, audios = zip(*sorted_items)

        lengths = [b.shape[0] for b in blendshapes]
        padded_blendshapes = pad_sequence(blendshapes, batch_first=True, padding_value=0.0)  # [B, T_max, 58]

        blendshape_mask = torch.zeros(padded_blendshapes.shape[:2], dtype=torch.bool)
        for i, l in enumerate(lengths):
            blendshape_mask[i, :l] = 1

        # Pad audio to max length
        audio_lengths = [a.shape[-1] for a in audios]  # [1, 640*T]
        max_audio_len = max(audio_lengths)
        padded_audios = torch.zeros((len(audios), 1, max_audio_len))

        audio_mask = torch.zeros((len(audios), max_audio_len), dtype=torch.bool)
        for i, a in enumerate(audios):
            l = a.shape[-1]
            padded_audios[i, 0, :l] = a
            audio_mask[i, :l] = 1

        return padded_blendshapes, blendshape_mask, padded_audios, audio_mask

    else:  # Only blendshapes
        blendshapes = batch
        blendshapes = sorted(blendshapes, key=lambda x: x.shape[0], reverse=True)
        lengths = [x.shape[0] for x in blendshapes]
        padded_blendshapes = pad_sequence(blendshapes, batch_first=True, padding_value=0.0)

        blendshape_mask = torch.zeros(padded_blendshapes.shape[:2], dtype=torch.bool)
        for i, l in enumerate(lengths):
            blendshape_mask[i, :l] = 1

        return padded_blendshapes, blendshape_mask



def get_dataloaders(args, test_config=False):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args, test_config)

    if not test_config:
        train_data = Dataset(train_data, subjects_dict, "train", args.read_audio)

        dataset["train"] = data.DataLoader( dataset=train_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            collate_fn=collate_fn_audio,
                                            drop_last=True
                                            )
        
        valid_data = Dataset(valid_data, subjects_dict, "val", args.read_audio)

        dataset["valid"] = data.DataLoader( dataset=valid_data, 
                                            batch_size=1, 
                                            shuffle=False, 
                                            num_workers=args.workers, 
                                            collate_fn=collate_fn_audio,
                                            drop_last=True)

    test_data = Dataset(test_data, subjects_dict, "test", args.read_audio)

    dataset["test"] = data.DataLoader( dataset=test_data, 
                                       batch_size=1, 
                                       shuffle=True, 
                                       num_workers=args.workers, 
                                       collate_fn=collate_fn_audio,
                                       drop_last=True)

    return dataset


if __name__ == "__main__":
    get_dataloaders()