python3 ./train/train_denoise.py --arch Uformer_B --batch_size 4 --gpu '0' \
    --train_ps 128 --roor_dir /mnt/d/fyq/sinogram/2e9div_smooth --env _0706\
      --save_dir ./logs/ --dataset sidd --warmup