CUDA_VISIBLE_DEVICES=0 \
python groupon_hsnet/test.py \
                  --backbone resnet50 \
                  --fold 0 \
                  --benchmark pascal \
                  --nshot 1 \
                  --load "groupon_hsnet/logs/checkpoint.pt" \
                  --agg 'mean' \
                  --N_q 2 \
                #   --visualize