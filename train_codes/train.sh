export VAE_MODEL="../models/sd-vae-ft-mse/"
export DATASET="../data"
# export UNET_CONFIG="../models/musetalk/musetalk.json"
export UNET_CONFIG="../models/musetalk/musetalk_mid3D.json"

accelerate launch --num_processes=1 train_video.py \
--mixed_precision="fp16" \
--unet_config_file=$UNET_CONFIG \
--pretrained_model_name_or_path=$VAE_MODEL \
--data_root=$DATASET \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=100000 \
--learning_rate=5e-05 \
--max_grad_norm=1 \
--lr_warmup_steps=0 \
--output_dir="3d_general" \
--val_out_dir='val' \
--testing_speed \
--checkpointing_steps=10000 \
--validation_steps=1000 \
--reconstruction \
--use_audio_length_left=2 \
--use_audio_length_right=2 \
--whisper_model_type="tiny" \
--train_json="../train.json" \
--val_json="../test.json" \
--lr_scheduler="cosine" \
--initial_unet_checkpoint="/data/scene-rep/u/michalstary/challenge/MuseTalk/models/musetalk/pytorch_model.bin" \
--freeze_spatial_unet \
--sequence_length=16 \
# --resume_from_checkpoint="latest" \
