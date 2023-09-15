PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=2
MAX_STEPS=1500
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /code/ptuning_data/talk_ct_new_wtxq/talk_ct_new_wtxq_train.json \
    --validation_file /code/ptuning_data/talk_ct_new_wtxq/talk_ct_new_wtxq_test.json \
    --preprocessing_num_workers 1 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /code/chatglm2-6b \
    --output_dir output/talk_ct_new_wtxq-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR-$MAX_STEPS-384-256 \
    --overwrite_output_dir \
    --max_source_length 384 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps $MAX_STEPS \
    --logging_steps 30 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    #--quantization_bit 4

