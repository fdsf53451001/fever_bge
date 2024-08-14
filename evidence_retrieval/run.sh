# python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
# --model_name_or_path BAAI/bge-large-zh-v1.5 \
# --input_file ./pre_train.jsonl \
# --output_file ./pre_train_hn_mine.jsonl \
# --range_for_sampling 2-10 \
# --use_gpu_for_searching

torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir "./checkpoint" \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data ./contrastivePreprocessed/train_evi.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 3 \
--save_total_limit 3 \
--per_device_train_batch_size 2 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 128 \ 
--passage_max_len 512 \ 
--train_group_size 16 \
--use_inbatch_neg False \
--logging_steps 10 \
--query_instruction_for_retrieval "" 

# python -m FlagEmbedding.baai_general_embedding.finetune.test \
# --train_data ./contrastivePreprocessed/train.jsonl \
# --model_name_or_path BAAI/bge-large-en-v1.5 \
# --output_dir "./test" 