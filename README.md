# KoBERT-KLUE

- KLUE (Korean Language Understanding Evaluation) with KoBERT
- 🤗Huggingface Tranformers🤗 라이브러리를 이용하여 구현

## Dependencies

- torch==1.9.1
- transformers==4.11.0
- tensorboardX>=2.0

### 1. Training

```bash
$ python3 run_klue.py  --model_type kobert \
                       --model_name_or_path monologg/kobert \
                       --output_dir models \
                       --data_dir data \
                       --train_file KorQuAD_v1.0_train.json \
                       --predict_file KorQuAD_v1.0_dev.json \
                       --evaluate_during_training \
                       --per_gpu_train_batch_size 8 \
                       --per_gpu_eval_batch_size 8 \
                       --max_seq_length 512 \
                       --logging_steps 4000 \
                       --save_steps 4000 \
                       --do_train
```

### 2. Evaluation

```console
$ python3 evaluate_v1_0.py {$data_dir}/KorQuAD_v1.0_dev.json {$output_dir}/predictions_.json
```

## Results


## References

- [KLUE Baseline](https://github.com/KLUE-benchmark/KLUE-baseline)
- [KLUE](https://klue-benchmark.com/)
- [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
