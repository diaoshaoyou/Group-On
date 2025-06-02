CUDA_VISIBLE_DEVICES=2 \
python groupon_hsnet/test.py \
                  --backbone resnet50 \
                  --fold 0 \
                  --benchmark pascal \
                  --nshot 1 \
                  --load "groupon_hsnet/logs/f0_pascal_r50/best_model.pt" \
                  --agg 'mean' \
                  --N_q 2 \
                  --visualize