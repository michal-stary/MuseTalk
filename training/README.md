# Data preprocessing

Create two config yaml files, one for training and other for testing (both in same format as configs/inference/test.yaml)
The train yaml file should contain the training video paths and corresponding audio paths
The test yaml file should contain the validation video paths and corresponding audio paths

Run:
```
cd REPOSITORY_ROOT
./preprocessing/preprocess_data.sh train output $(ls -d data/video/downloaded_videos/*)
./preprocessing/preprocess_data.sh test output val_video1 val_video2
```
This creates folders which contain the image frames and npy files. This also creates train.json and val.json which can be used during the training.

## Data organization
```
./data/
├── images
│     └──RD_Radio10_000
│         └── 0.png
│         └── 1.png
│         └── xxx.png
│     └──RD_Radio11_000
│         └── 0.png
│         └── 1.png
│         └── xxx.png
├── audios
│     └──RD_Radio10_000
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
│     └──RD_Radio11_000
│         └── 0.npy
│         └── 1.npy
│         └── xxx.npy
```

## Training
Simply run after preparing the preprocessed data
```
cd REPOSITORY_ROOT/training
sh train.sh
```
