from klue_baseline.metrics.functional import *

def metric_ynat(preds, targets):
    return {'macro_f1': ynat_macro_f1(preds, targets)}


def metric_nli(preds, targets):
    return {'accuracy': klue_nli_acc(preds, targets)}


def metric_sts(preds, targets):
    return {'pearsonr': klue_sts_pearsonr(preds, targets),
    'f1':klue_sts_f1(preds, targets)}


def metric_re(probs, preds, targets):
    return {'f1': klue_re_micro_f1(preds, targets),
    'auprc':klue_re_auprc(probs, targets)}


metrics = {
    'ynat': metric_ynat,
    'nli' : metric_nli,
    'sts' : metric_sts,
    're' : metric_re
}