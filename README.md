# KoBERT-KLUE

- KLUE (Korean Language Understanding Evaluation) with KoBERT
- 🤗Huggingface Tranformers🤗 라이브러리를 이용하여 구현

## Dependencies

- torch==1.9.1
- transformers==4.11.0
- tensorboardX>=2.0

### Training

```bash
$ python3 run_klue.py ynat
```

## Results
```
ynat task
macro_f1 86.44

nli task
accuracy 80.2

sts task
pearsonr 68.65
f1 82.62

re task
f1 67.10
auprc 69.73

ner task
char_macro_f1 93.01
entity_macro_f1 87.76
```

## References

- [KLUE Baseline](https://github.com/KLUE-benchmark/KLUE-baseline)
- [KLUE](https://klue-benchmark.com/)
- [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
