import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def read_video_urls(file_path):
    """Read video URLs and names from the input file."""
    videos = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                name, url = line.strip().split(' ', 1)
                videos[name] = url.strip()
    return videos

def download_video(name, url, output_dir):
    """Download a single video in 720p resolution (or best available below 720p)."""
    output_template = os.path.join(output_dir, f"{name}.mp4")
    
    # Skip if the file already exists
    if os.path.exists(output_template):
        return f"Skipped {name} (already exists)"
    
    try:
        # Use yt-dlp to download the video
        command = [
            'yt-dlp',
            '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]',  # Select 720p or lower
            '-o', output_template,
            '--merge-output-format', 'mp4',
            url
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        return f"Successfully downloaded {name}"
    except subprocess.CalledProcessError as e:
        return f"Error downloading {name}: {str(e)}"

def main():
    # Create output directory
    output_dir = "../data/video/downloaded_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read video URLs
    videos = read_video_urls("WDA_video_url.txt")
    
    # Download videos in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for name, url in videos.items():
            future = executor.submit(download_video, name, url, output_dir)
            futures.append(future)
        
        # Show progress bar
        for future in tqdm(futures, total=len(futures), desc="Downloading videos"):
            result = future.result()
            print(result)

if __name__ == "__main__":
    main() 