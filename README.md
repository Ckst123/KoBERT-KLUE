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
macro_f1 : 85.96

nli task
accuracy 69.90

stc task
pearsonr 65.39
f1 80.90

re
f1 58.37967834059823
auprc 57.644041712517335
```

## References

- [KLUE Baseline](https://github.com/KLUE-benchmark/KLUE-baseline)
- [KLUE](https://klue-benchmark.com/)
- [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
