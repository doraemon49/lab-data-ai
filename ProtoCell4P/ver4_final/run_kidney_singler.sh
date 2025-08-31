python run.py \
  --data split_datasets \
  --data_location /data/project/kim89/0819_kidney \
  --tasks kidney \
  --load_ct \
  --cell_type_annotation singler_annotation \
  --n_repeats 5 \
  --n_folds 5 \
  --tune \
  --n_trials 3 --top_k 1 \
  --device cuda:3 \
  --batch_size 512 \
  --exp_str kidney_singler \
  --model ProtoCell \
  --pretrained --max_epoch_pretrain 75 \
  > logs/kidney/singler_annotation.txt 2>&1
