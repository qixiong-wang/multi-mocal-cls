CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 1 \
--master_port 12335 train.py --model lavt  --batch-size 8 --lr 0.00005 --wd 1e-2 \
--swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 40 --output-dir checkpoints_text_512 --img_size 480 2>&1 | tee ./logs/lavt_base_480_text_512.log
