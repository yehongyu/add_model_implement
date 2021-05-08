from sklearn import metrics

def aucfun(act, pred):
    fpr, tpr, thresholds = metrics.roc_curve(act, pred, pos_label=1)
    print fpr, tpr, thresholds
    return metrics.auc(fpr, tpr)


if __name__ == '__main__':
    act = [1, 1, 0, 0, 1]
    pred = [0.5, 0.6, 0.55, 0.4, 0.7]
    aucfun(act, pred)