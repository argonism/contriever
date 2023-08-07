
python finetuning.py \
    --train_data dataset/mr-tydi/ja/train.jsonl \
    --eval_data dataset/mr-tydi/ja/dev.jsonl \
    --model_path facebook/mcontriever-msmarco \
    --retriever_model_id bert-base-multilingual-cased \
    --continue_training \
    --num_workers 1 \
    --ratio_min 0.1 --ratio_max 0.5 \
    --temperature 0.05 \
    --lr 0.00005 \
    --warmup_steps 20000 \
    --scheduler linear \
    --optim adamw \
    --per_gpu_batch_size 32 \
    --output_dir model/mrtidy_jp \
    --total_steps 20000 \
    --save_freq 1000 \
    --negative_ctxs 1 \
    --chunk_length 256

python finetuning.py \
    --train_data dataset/ntcir-transfer/1/train.jsonl \
    --eval_data dataset/ntcir-transfer/1/dev.jsonl \
    --model_path facebook/mcontriever-msmarco \
    --retriever_model_id bert-base-multilingual-cased \
    --continue_training \
    --num_workers 1 \
    --temperature 0.05 \
    --lr 0.00005 \
    --warmup_steps 20000 \
    --scheduler linear \
    --optim adamw \
    --per_gpu_batch_size 32 \
    --output_dir model/transfer \
    --total_steps 20000 \
    --save_freq 1000 \
    --negative_ctxs 1


python -m tevatron.driver.train \
  --output_dir model_ntcir_tevatron \
  --model_name_or_path facebook/mcontriever-msmarco \
  --save_steps 1 \
  --dataset_name json \
  --train_dir dataset/ntcir-transfer/ja/train.jsonl \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 8 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 40 \
  --grad_cache \
  --gc_p_chunk_size 8 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --eval_steps 1 \
  --do_eval

python train_with_tevatron.py \
  --output_dir model_mrtydi_japanese \
  --model_name_or_path facebook/mcontriever-msmarco \
  --save_steps 1 \
  --dataset_name json \
  --train_dir dataset/ntcir-transfer/ja/train.jsonl \
  --fp16 \
  --per_device_train_batch_size 1024 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 10 \
  --grad_cache \
  --gc_p_chunk_size 8 \
  --logging_steps 1 \
  --overwrite_output_dir
