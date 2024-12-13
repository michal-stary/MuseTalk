import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import uuid
from omegaconf import OmegaConf
import argparse
import pickle
from musetalk.utils import hack_registry
from musetalk.utils.utils import load_all_model, get_video_fps, get_file_type
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
import glob
import shutil

def get_next_index(folder_path):
    """Returns the next available index in the folder."""
    if not os.path.isdir(folder_path):
        return 0
    files = [int(os.path.splitext(f)[0]) for f in os.listdir(folder_path) if f[0].isdigit()]
    return max(files, default=-1) + 1

def process_video_frames(video_path, bbox_shift, use_saved_coord=False, result_dir=None, fps=None):
    """Extract and process frames from video source."""
    # Handle different input types (video file, image, directory)
    if get_file_type(video_path) == "video":
        save_dir = os.path.join("temp", os.path.basename(video_path).split('.')[0])
        os.makedirs(save_dir, exist_ok=True)
        os.system(f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir}/%08d.png")
        input_imgs = sorted(glob.glob(os.path.join(save_dir, '*.[jpJP][pnPN]*[gG]')))
        fps = fps or get_video_fps(video_path)
    elif get_file_type(video_path) == "image":
        input_imgs = [video_path]
        fps = fps or 25
    elif os.path.isdir(video_path):
        input_imgs = sorted(glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]')))
        fps = fps or 25
    else:
        raise ValueError(f"Invalid input: {video_path}")

    # Handle saved coordinates if enabled
    input_basename = os.path.basename(video_path).split('.')[0]
    coord_path = os.path.join(result_dir, input_basename, f"{input_basename}_landmarks.pkl") if result_dir else None
    
    if use_saved_coord and coord_path and os.path.exists(coord_path):
        print(f"Using saved landmarks from {coord_path}")
        with open(coord_path, 'rb') as f:
            coords = pickle.load(f)
        frames = read_imgs(input_imgs)
    else:
        print("Extracting landmarks... (this may take a while)")
        coords, frames = get_landmark_and_bbox(input_imgs, bbox_shift)
        
        if coord_path:
            save_dir = os.path.dirname(coord_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f)
            print(f"Saved landmarks to {coord_path}")

    # Process frames to get cropped versions
    processed_frames = []
    for bbox, frame in zip(coords, frames):
        if bbox == coord_placeholder:
            continue
            
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
        
        if (y2-y1) <= 0 or (x2-x1) <= 0:
            continue
            
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        processed_frames.append(crop_frame)

    # Cleanup temporary directory if it was created
    if get_file_type(video_path) == "video" and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    return processed_frames, coords, fps

def save_processed_data(frames, coords, audio_chunks, output_dir, 
                       start_img_idx, start_audio_idx, audio_processor=None):
    """Save processed frames and audio features."""
    os.makedirs(os.path.join("data/images", output_dir, ), exist_ok=True)
    os.makedirs(os.path.join("data/audios", output_dir, ), exist_ok=True)
    
    print(f"Processing {len(frames)} frames")
    print(f"Processing {len(audio_chunks)} audio chunks")
    if frames and audio_chunks:
        print(f"Frame shape: {frames[0].shape}")
        print(f"Audio chunks shape: {audio_chunks[0].shape}")

    img_idx = start_img_idx
    audio_idx = start_audio_idx
    
    # Save frames
    for frame in frames:
        if frame is not None:
            cv2.imwrite(f"data/images/{output_dir}/{img_idx}.png", frame)
            img_idx += 1
        else:
            print("Warning: Some frames are None, audio may not be aligned")
            
    # Save audio chunks
    for chunk in audio_chunks:
        np.save(f"data/audios/{output_dir}{audio_idx}.npy", chunk)
        audio_idx += 1
    
    return img_idx, audio_idx

@torch.no_grad()
def main(args):
    # Load models
    audio_processor, _, _, _ = load_all_model()
    
    # Create output directories
    output_dir = f"{args.folder_name}"
    os.makedirs("data/images/" + output_dir, exist_ok=True)
    os.makedirs("data/audios/" + output_dir, exist_ok=True)
    # Get starting indices
    start_img_idx = get_next_index(f"data/images/{output_dir}")
    start_audio_idx = get_next_index(f"data/audios/{output_dir}")
    
    # Process each task in config
    config = OmegaConf.load(args.inference_config)
    for task_id in config:
        task = config[task_id]
        
        # Process video frames
        frames, coords, fps = process_video_frames(
            task["video_path"],
            task.get("bbox_shift", args.bbox_shift),
            use_saved_coord=args.use_saved_coord,
            result_dir=args.result_dir,
        )
        
        # Process audio
        audio_features = audio_processor.audio2feat(task["audio_path"]) # 50 FPS audio features
        
        print("audio feat", audio_features.shape)
        print(f"Video FPS: {fps}")
        print(len(frames), len(audio_features))

        # align the fps of audio and video
        audio_chunks = audio_processor.feature2chunks(audio_features, fps, audio_feat_length=[2,2])
        
        # Save processed data
        start_img_idx, start_audio_idx = save_processed_data(
            frames, coords, audio_chunks,
            output_dir, start_img_idx, start_audio_idx, audio_processor=audio_processor
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--folder_name", default=str(uuid.uuid4()))
    parser.add_argument("--use_saved_coord", action="store_true", help="Use saved landmarks if available")
    parser.add_argument("--result_dir", default="./results", help="Directory to save/load landmarks")
    args = parser.parse_args()
    main(args)
