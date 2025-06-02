CUDA_VISIBLE_DEVICES=0 \
python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=19002 \
groupon_hsnet/train.py \
                  --backbone resnet50 \
                  --fold 0 \
                  --benchmark pascal \
                  --lr 1e-3 \
                  --bsz 20 \
                  --logpath "" \
                  --load "" \
                  --group_start 10 \
                  --N_q 2