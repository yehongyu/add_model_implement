#coding:utf-8
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb


def load(model_path):
    model = xgb.Booster({'nthread': 8})
    model.load_model(model_path)
    return model


def read(libsvm_path, num_features, feature_start_idx=1, slot_mask=None):
    if slot_mask is None:
        slot_mask = []
    else:
        slot_mask = map(int, slot_mask.split(','))
    labels = []
    features = []
    with open(libsvm_path, 'r') as fin:
        for line in fin.readlines():
            row = [np.nan] * num_features
            fields = line.split(' ')
            label = int(fields[0])
            assert(label == 0 or label == 1)
            labels.append(label)
            for idx in range(feature_start_idx, len(fields)):
                pair = fields[idx].split(':')
                slot = int(pair[0])
                if slot in slot_mask:
                    continue
                assert(np.isnan(row[slot])) # Forbid duplicate slots
                row[slot] = float(pair[1])
            features.append(row)
    return np.array(labels), np.array(features)


def evaluate(df, threshold=0.5):
    TP = len(df[(df['score'] >= threshold) & (df['label'] == 1)])
    Ttotal = len(df[df['score'] >= threshold])
    Ptotal = len(df[df['label'] == 1])
    precision = 1.0 * TP / max(Ttotal, 1)
    recall = 1.0 * TP / max(Ptotal, 1)
    f1 = 2.0 * precision * recall / max(1, precision + recall)
    share = Ttotal * 1.0 / len(df)
    return {
        'threshold': threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'share': share
    }


def main(FLAGS):
    labels, features = read(
        FLAGS.input_path, FLAGS.num_features, FLAGS.feature_start_column, FLAGS.slot_mask)
    input = xgb.DMatrix(features, missing=np.nan)
    scores = load(FLAGS.model_path).predict(input)
    df = pd.DataFrame({'score': scores, 'label': labels})
    threshold_list = map(float, FLAGS.threshold.split(','))
    for threshold in threshold_list:
        metrics = evaluate(df, threshold)
        print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Predict and evaluate a binary XGBoost model at local.')
    parser.add_argument('--feature-start-column',
        type=int, default=1,
        help='Which index the feature columns start from in LibSVM.')
    parser.add_argument('--input-path',
        type=str, required=True,
        help='Path to the LibSVM file for evaluation.')
    parser.add_argument('--model-path',
        type=str, required=True,
        help='Path to a pretrained XGBoost model.')
    parser.add_argument('--num-features',
        type=int,
        help='Count of features in the model.')
    parser.add_argument('--threshold',
        default='0.01',
        help='Which score to distinguish instances as positive or negative.')
    parser.add_argument('--slot-mask',
        help='Slot masks, for example 1,2,4,5')
    FLAGS = parser.parse_args()
    main(FLAGS)
