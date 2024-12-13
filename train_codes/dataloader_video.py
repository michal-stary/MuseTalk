import os
import cv2
import torch
import numpy as np
from torch.utils import data as data_utils
from os.path import dirname, join, basename, isfile
from glob import glob
from tqdm import tqdm
import json

from utils.utils import prepare_mask_and_masked_image

class VideoDataset(data_utils.Dataset):
    """Dataset for loading sequences of video frames and corresponding audio features.
    
    This dataset loads sequences of video frames and their corresponding whisper audio features,
    returning them in a deterministic, sequential order. Each sequence contains consecutive frames
    from a video, with their reference frames and audio context.
    """
    
    SYNC_NET_T = 1  # Number of frames to sync
    IMG_SIZE = 256  # Size to resize images to
    
    def __init__(self, 
                 data_root: str,
                 json_path: str,
                 sequence_length: int = 32,  # Length of frame sequences to return
                 use_audio_length_left: int = 1,
                 use_audio_length_right: int = 1,
                 whisper_model_type: str = "tiny"):
        """Initialize the dataset.
        
        Args:
            data_root: Root directory containing the video frames
            json_path: Path to JSON file containing video paths
            sequence_length: Number of consecutive frames to return in each sequence
            use_audio_length_left: Number of audio frames to use from before current frame
            use_audio_length_right: Number of audio frames to use from after current frame
            whisper_model_type: Type of whisper model used ("tiny" or "largeV2")
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.audio_context = (use_audio_length_left, use_audio_length_right)
        self.img_names_path = '../data'
        self.whisper_model_type = whisper_model_type
        
        # Set up whisper model parameters
        if whisper_model_type == "tiny":
            self.whisper_path = '../data/audios'
            self.whisper_dims = (5, 384)  # (W, H)
        elif whisper_model_type == "largeV2":
            self.whisper_path = '...'
            self.whisper_dims = (33, 1280)
        else:
            raise ValueError(f"Unsupported whisper model type: {whisper_model_type}")
            
        # Calculate total width of concatenated audio features
        self.whisper_feature_total_width = (
            self.whisper_dims[0] * 2 * (use_audio_length_left + use_audio_length_right + 1)
        )
        
        # Load video paths and prepare frame lists
        self.videos = self._load_videos(json_path)
        self.frame_lists = self._prepare_frame_lists()
        
        # Calculate valid sequences for each video
        self.sequence_starts = self._calculate_sequence_starts()
        
    def _load_videos(self, json_path: str) -> list:
        """Load video paths from JSON file."""
        with open(json_path, 'r') as file:
            return json.load(file)
            
    def _prepare_frame_lists(self) -> list:
        """Prepare lists of frame paths for each video."""
        frame_lists = []
        for vidname in tqdm(self.videos, desc="Preparing dataset"):
            json_path = f"{self.img_names_path}/{vidname.split('/')[-1].split('.')[0]}.json"
            
            if not os.path.exists(json_path):
                # Create new JSON with sorted frame paths
                img_names = sorted(
                    glob(join(vidname, '*.png')),
                    key=lambda x: int(x.split("/")[-1].split('.')[0])
                )
                with open(json_path, "w") as f:
                    json.dump(img_names, f)
            else:
                # Load existing frame list
                with open(json_path, "r") as f:
                    img_names = json.load(f)
                    
            frame_lists.append(img_names)
        return frame_lists
        
    def _get_frame_id(self, frame_path: str) -> int:
        """Extract frame ID from frame path."""
        return int(basename(frame_path).split('.')[0])
        
    def _get_window(self, start_frame: str) -> list:
        """Get list of frame paths for a temporal window."""
        start_id = self._get_frame_id(start_frame)
        vidname = dirname(start_frame)
        
        window_fnames = []
        for frame_id in range(start_id, start_id + self.SYNC_NET_T):
            frame = join(vidname, f'{frame_id}.png')
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames
        
    def _read_window(self, window_fnames: list) -> np.ndarray:
        """Read and preprocess a window of frames."""
        if window_fnames is None:
            return None
            
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
                
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                window.append(img)
            except Exception as e:
                print(f"Error processing frame {fname}: {e}")
                return None
                
        return window
        
    def _prepare_window(self, window: list) -> np.ndarray:
        """Convert window of frames to proper format."""
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x
        
    def _load_audio_features(self, vidname: str, frame_idx: int) -> torch.Tensor:
        """Load and concatenate audio features for a frame."""
        sub_folder_name = vidname.split('/')[-1]
        audio_dir = os.path.join(self.whisper_path, sub_folder_name)
        
        if not os.path.isdir(audio_dir):
            raise ValueError(f"Audio directory not found: {sub_folder_name}")
            
        # Load features for the temporal window
        features = []
        start_idx = frame_idx - self.audio_context[0]
        end_idx = frame_idx + self.audio_context[1] + 1
        
        for feat_idx in range(start_idx, end_idx):
            feat_path = os.path.join(audio_dir, f"{feat_idx}.npy")
            if not os.path.exists(feat_path):
                raise ValueError(f"Missing audio feature: {feat_path}")
                
            try:
                features.append(np.load(feat_path))
            except Exception as e:
                raise ValueError(f"Error loading audio feature {feat_path}: {str(e)}")
                
        # Concatenate and reshape
        audio_feature = np.concatenate(features, axis=0)
        audio_feature = audio_feature.reshape(1, -1, self.whisper_dims[1])
        
        expected_shape = (1, self.whisper_feature_total_width, self.whisper_dims[1])
        if audio_feature.shape != expected_shape:
            raise ValueError(
                f"Invalid audio feature shape for {vidname} {frame_idx}: "
                f"got {audio_feature.shape}, expected {expected_shape}"
            )
            
        return torch.squeeze(torch.FloatTensor(audio_feature))
        
    def _calculate_sequence_starts(self) -> list:
        """Calculate valid starting points for sequences in each video."""
        sequence_starts = []
        min_context = max(5, self.audio_context[0])
        
        for video_frames in self.frame_lists:
            # Calculate how many complete sequences we can get from this video
            valid_frames = len(video_frames) - min_context
            if valid_frames <= 0:
                sequence_starts.append([])
                continue
                
            # Get all possible sequence starting points
            starts = []
            current_start = min_context
            while current_start < len(video_frames):
                remaining_frames = len(video_frames) - current_start
                if remaining_frames > 0:  # Accept any sequence length > 0
                    starts.append(current_start)
                current_start += self.sequence_length
                
            sequence_starts.append(starts)
            
        return sequence_starts
        
    def __len__(self) -> int:
        """Return total number of sequences across all videos."""
        return sum(len(starts) for starts in self.sequence_starts)
        
    def __getitem__(self, idx: int) -> tuple:
        """Get a sequence of frames with their reference frames and audio features.
        
        Args:
            idx: Global sequence index
            
        Returns:
            tuple: (reference_frames, target_frames, masked_targets, masks, audio_features)
            Each element is a list/tensor of length up to sequence_length
        """
        # Find which video and sequence this index corresponds to
        video_idx = 0
        seq_offset = idx
        
        for starts in self.sequence_starts:
            if seq_offset >= len(starts):
                seq_offset -= len(starts)
                video_idx += 1
            else:
                break
                
        if video_idx >= len(self.videos):
            raise IndexError("Index out of range")
            
        # Get video information
        vidname = self.videos[video_idx].split('/')[-1]
        video_frames = self.frame_lists[video_idx] 
        start_idx = self.sequence_starts[video_idx][seq_offset]
        
        # Determine actual sequence length (might be shorter at end of video)
        actual_length = min(self.sequence_length, (len(video_frames) - 11) - start_idx)
        
        # Initialize lists for sequence data 
        ref_images = []
        images = []
        masked_images = []
        masks = []
        audio_features = []
        
        # Process each frame in the sequence
        for i in range(actual_length):
            frame_idx = start_idx + i
            target_frame = video_frames[frame_idx]
            ref_frame = video_frames[max(0, frame_idx - 5)]  # Reference frame is 5 frames before
            
            # Load frame windows
            target_window = self._get_window(target_frame)
            ref_window = self._get_window(ref_frame)
            
            if target_window is None or ref_window is None:
                raise ValueError(f"Missing frames for video {vidname} at index {frame_idx}")
                
            # Process frames
            target_frames = self._read_window(target_window)
            ref_frames = self._read_window(ref_window)
            
            if target_frames is None or ref_frames is None:
                raise ValueError(f"Error reading frames for video {vidname} at index {frame_idx}")
                
            # Prepare images
            target_window = self._prepare_window(target_frames)
            image = target_window.copy().squeeze()
            
            # Create mask (upper half face)
            target_window[:, :, target_window.shape[2]//2:] = 0.
            ref_image = self._prepare_window(ref_frames).squeeze()
            
            # Prepare mask and masked image
            mask = torch.zeros((ref_image.shape[1], ref_image.shape[2]))
            mask[:ref_image.shape[2]//2, :] = 1
            image = torch.FloatTensor(image)
            mask, masked_image = prepare_mask_and_masked_image(image, mask)
            
            # Load audio features
            audio_feature = self._load_audio_features(vidname, self._get_frame_id(target_frame))
            
            # Add to sequence lists
            ref_images.append(ref_image)
            images.append(image)
            masked_images.append(masked_image)
            masks.append(mask)
            audio_features.append(audio_feature)
        
        # Stack all arrays in the sequence using numpy
        ref_images = np.stack(ref_images)
        images = np.stack(images)
        masked_images = np.stack(masked_images)
        masks = np.stack(masks)
        audio_features = np.stack(audio_features)
        
        return ref_images, images, masked_images, masks, audio_features
         
    
    
if __name__ == "__main__":
    data_root = '..'
    val_data = VideoDataset(data_root, 
                          '/data/scene-rep/u/michalstary/challenge/MuseTalk/test.json', 
                          use_audio_length_left = 2,
                          use_audio_length_right = 2,
                          whisper_model_type = "tiny"
                          )
    val_data_loader = data_utils.DataLoader(
        val_data, batch_size=1, shuffle=True,
        num_workers=1)
    

    for i, data in enumerate(val_data_loader):
        ref_image, image, masked_image, mask, audio_feature = data
        print(i)

            
 
