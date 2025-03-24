python3 ./train/train_sinogram.py --arch Uformer_B --batch_size 4 --gpu '0' \
    --train_ps 128 --root_dir /mnt/d/fyq/sinogram/2e9div_smooth --env _0706\
      --save_dir ./logs/ --dataset sidd --warmup --embed_dim 8 --vit_dim 64 --vit_depth 4\
      --vit_nheads 4 --vit_mlp_dim 128