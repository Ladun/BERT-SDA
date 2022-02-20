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
                         --do_train
```

