# 

```
python run_classifier.py --model_type bert \
                         --model_name_or_path bert-base-multilingual-cased \
                         --output_dir outputs \
                         --dataset_name snli \
                         --per_gpu_train_batch_size 8 \
                         --per_gpu_eval_batch_size 8 \
                         --max_seq_length 512 \
                         --logging_steps 4000 \
                         --save_steps 4000 \
                         --evaluate_during_training \
                         --do_train
```
 
02/28/2022 05:53:46 - INFO - __main__ -   per_gpu_train_batch_size: 8
02/28/2022 05:53:46 - INFO - __main__ -   per_gpu_eval_batch_size: 8
02/28/2022 05:53:46 - INFO - __main__ -   max_seq_length: 128
02/28/2022 05:53:46 - INFO - __main__ -   learning_rate: 5e-05
02/28/2022 05:53:46 - INFO - __main__ -   gradient_accumulation_steps: 1
02/28/2022 05:53:46 - INFO - __main__ -   weight_decay: 0.0
02/28/2022 05:53:46 - INFO - __main__ -   adam_epsilon: 1e-08
02/28/2022 05:53:46 - INFO - __main__ -   max_grad_norm: 1.0
02/28/2022 05:53:46 - INFO - __main__ -   num_train_epochs: 3.0

02/28/2022 05:53:46 - INFO - __main__ -   seed: 2022



02/28/2022 05:53:46 - INFO - __main__ -   distillation_type: average
02/28/2022 05:53:46 - INFO - __main__ -   kd_lambda: 1.0
02/28/2022 05:53:46 - INFO - __main__ -   kd_K: 1
=> Results: {'acc': 0.870928338762215, 'loss': 0.6886479279894006}