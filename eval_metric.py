from klue_baseline.metrics.functional import *

def metric_ynat(preds, targets):
    return {'macro_f1': ynat_macro_f1(preds, targets)}

metrics = {'ynat': metric_ynat}