export CUDA_VISIBLE_DEVICES=0,1,2,3
for((i=0;i<5;i++));  
do   

python run_xlnet.py \
--model_type xlnet \
--model_name_or_path ./chinese_xlnet_mid \
--do_train \
--do_eval \
--do_test \
--data_dir ./data/data_$i \
--output_dir ./model_xlnet$i \
--max_seq_length 150 \
--split_num 10 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 64 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 4000 \
--report_steps 200 ;

done  





