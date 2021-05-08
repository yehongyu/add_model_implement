#!/usr/bin/env python
#coding=utf8
from __future__ import print_function

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s'
)

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'cpp'

import json
import numpy as np
import random
import struct
import sys

from collections import defaultdict, namedtuple
from tensorflow.core.example import example_pb2, feature_pb2


SearchPair = namedtuple('SearchPair', ['query', 'label_box'])
Document = namedtuple('Document', [
    'tagtitle', 'realtitle', 'maintitle', 'anchor', 'host', 'cq', 'feature_str'
])


def _bytes_feature(values):
    if not isinstance(values, list):
        values = [values]
    return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=values))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[value]))

def _int64_feature(values):
    if not isinstance(values, list):
        values = [values]
    return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=values))


def parse_one_instance(line):
    data = json.loads(line.strip())
    rs = data.get('rs', [])
    query = data.get('query', '')

    # parse json fields
    pos_sample_num = 0
    label_box = defaultdict(lambda: [])
    for key in rs:
        fields = rs[key].split('\t')
        if len(fields) != 7:
            continue

        # cq & label
        cq_label = fields[6]
        cq_label_split = cq_label.split(' ')
        cq = ' '.join(cq_label_split[0:-1])
        label = int(cq_label[-1])
        if label != 0:
            pos_sample_num += 1

        # features
        float_feature = fields[0]
        features = float_feature.split(' ')
        feature_str = ' '.join(features[0:-1])

        # reamined fields
        tagtitle = fields[1]
        realtitle = fields[2]
        maintitle = fields[3]
        anchor = fields[4]
        host = fields[5]

        # one record
        label_box[label].append(Document(
            tagtitle.encode('utf-8'),
            realtitle.encode('utf-8'),
            maintitle.encode('utf-8'),
            anchor.encode('utf-8'),
            host.encode('utf-8'),
            cq.encode('utf-8'),
            feature_str.encode('utf-8')
        ))

    # ensure a positive sample
    if pos_sample_num < 1:
        return None
    else:
        return SearchPair(query.encode('utf-8'), label_box)


def dump_tf_example(instance):
    label_box = instance.label_box
    query = instance.query
    example_list = []
    for pos_label in range(4, 1, -1):
        if pos_label not in label_box:
            continue
        pos_vec = label_box[pos_label]
        for neg_label in range(0, pos_label):
            if neg_label not in label_box:
                continue
            neg_vec = label_box[neg_label]
            if neg_label > 1 or len(pos_vec) >= len(neg_vec):
                num_pairs = min(len(pos_vec), min(3, len(neg_vec)))
                random_list = range(num_pairs)
            else:
                num_pairs = min(len(pos_vec), len(neg_vec))
                random_list = random.sample(range(0, len(neg_vec)), len(pos_vec))
            max_count = min(5, num_pairs)
            for pos_idx in range(max_count):
                pos_doc = pos_vec[pos_idx]
                neg_doc = neg_vec[random_list[pos_idx]]
                features_pb = feature_pb2.Features(feature={
                    'query': _bytes_feature(query),
                    'pos_tagtitle': _bytes_feature(pos_doc.tagtitle),
                    'neg_tagtitle': _bytes_feature(neg_doc.tagtitle),
                    'pos_realtitle': _bytes_feature(pos_doc.realtitle),
                    'neg_realtitle': _bytes_feature(neg_doc.realtitle),
                    'pos_maintitle': _bytes_feature(pos_doc.maintitle),
                    'neg_maintitle': _bytes_feature(neg_doc.maintitle),
                    'pos_anchor': _bytes_feature(pos_doc.anchor),
                    'neg_anchor': _bytes_feature(neg_doc.anchor),
                    'pos_host': _bytes_feature(pos_doc.host),
                    'neg_host': _bytes_feature(neg_doc.host),
                    'pos_cq': _bytes_feature(pos_doc.cq),
                    'neg_cq': _bytes_feature(neg_doc.cq),
                    'pos_feature': _bytes_feature(pos_doc.feature_str),
                    'neg_feature': _bytes_feature(neg_doc.feature_str),
                    'pos_label': _int64_feature(pos_label),
                    'neg_label': _int64_feature(neg_label)
                })
                example_list.append(example_pb2.Example(features=features_pb))
    return example_list


def write_tf_record(example):
    data = example.SerializeToString()
    data_size = struct.pack('<Q', len(data))
    fake_data = ''
    fake_data_size = struct.pack('<Q', len(fake_data))
    res = ''.join([data_size, data, fake_data_size, fake_data])
    print(res, end='')


def process_data(stream):
    count = 0
    for line in stream:
        try:
            instance = parse_one_instance(line)
            if instance:
                for item in dump_tf_example(instance):
                    write_tf_record(item)
                    count += 1
            logging.info(count)
        except Exception as e:
            logging.error('%s', e)


if __name__ == '__main__':
    process_data(sys.stdin)
