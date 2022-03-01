# BERT - SDA

A PyTorch implementation of "[Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation](https://arxiv.org/abs/2002.10345)"

## Dependencies

- torch == 1.8.3
- datasets == 1.18.3
- transformers == 4.7.0

## How to use

Currently, this repository only support snli datasets.

### 1. Training without Self-Ensemble and Self-Distillation
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
                         --do_train \
                         --do_eval
```


### 2. Training with Self-Ensemble and Self-Distillation

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
                         --do_train \
                         --do_eval \
                         --distillation_type 'average' \
                         --kd_K 1 \
```
 
## Result
|                           | Accuracy (%)  |
| ------------------------- | ------------- |
| bert-base-uncased (no kd) | 85.647        |
| bert-base-uncased (K = 1) | 87.092        |

<details>
<summary>Hyper Parameters </summary>

- Default parameters
    -   per_gpu_train_batch_size: 8
    -   per_gpu_eval_batch_size: 8
    -   max_seq_length: 128
    -   learning_rate: 5e-05
    -   gradient_accumulation_steps: 1
    -   weight_decay: 0.0
    -   adam_epsilon: 1e-08
    -   max_grad_norm: 1.0
    -   num_train_epochs: 3.0
    -   seed: 2022
- knowledge distillation parameters
    -   distillation_type: 'average'
    -   kd_lambda: 1.0
    -   kd_K:  >= 1


</details>

## Reference
- [BERT-SDA by lonePatient](https://github.com/lonePatient/BERT-SDA)