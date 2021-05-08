#coding:utf-8
from __future__ import print_function

import argparse
import pandas as pd


def load(label_and_score_path):
    ''' The result file should be of TSV with 2 columns in sequence: label, score.
    '''
    scores = []
    labels = []
    with open(label_and_score_path, 'r') as fin:
        for line in fin.readlines():
            fields = line.split('\t')
            label = int(fields[0].split('.', 1)[0])
            assert(label == 0 or label == 1)
            labels.append(label)
            scores.append(float(fields[1]))
    return pd.DataFrame({'score': scores, 'label': labels})


def evaluate(df, threshold=0.5):
    TP = len(df[(df['score'] >= threshold) & (df['label'] == 1)])
    Ttotal = len(df[df['score'] >= threshold])
    Ptotal = len(df[df['label'] == 1])
    precision = 1.0 * TP / max(Ttotal, 1)
    recall = 1.0 * TP / max(Ptotal, 1)
    f1 = 2.0 * precision * recall /max(1, precision + recall)
    share = Ttotal * 1.0 / len(df)
    return {
        'threshold': threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'share': share
    }


def main(FLAGS):
    df = load(FLAGS.output_path)
    metrics = evaluate(df, FLAGS.threshold)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Evaluate predicted result of a binary XGBoost model at local.')
    parser.add_argument('--output-path',
        type=str, default='reckon.evaluated',
        help='Path to the predicted result dumped from Reckon.')
    parser.add_argument('--threshold',
        type=float, default=0.01,
        help='Which score to distinguish instances as positive or negative.')
    FLAGS = parser.parse_args()
    main(FLAGS)
