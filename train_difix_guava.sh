# "--mixed_precision=bf16", 
python src/train_difix_guava.py \
    --seed=42 \
    --output_dir=outputs/difix/train \
    --dataset_path=/home/szj/err_empty_syn/ytb/bbox_square_x1.1/outputs/render_for_difix_v0.0.3-mix/difix_data.json \
    --max_train_steps=10000 \
    --resolution=512 \
    --learning_rate=2e-5 \
    --train_batch_size=1 \
    --dataloader_num_workers=8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 \
    --eval_freq=-1 \
    --viz_freq=100 \
    --lambda_lpips=1.0 \
    --lambda_l2=1.0 \
    --lambda_gram=1.0 \
    --gram_loss_warmup_steps=2000 \
    --report_to=wandb \
    --tracker_project_name=difix \
    --tracker_run_name=train_v0.0.3-mix \
    --timestep=199 \
    --mv_unet

