#!/usr/bin/env python
#coding=utf8
from __future__ import print_function

import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'cpp'

import sys
sys.path.insert(0, '.')

import numpy as np
import struct

from idl.matrix.proto import  proto_parser_pb2
from tensorflow.core.example import example_pb2, feature_pb2


def parse_ad_instance(stream):
    size_t = 8
    # kafka prefix
    stream.read(size_t*2)
    # size + sort_id
    size_binary = stream.read(size_t)[::-1]
    size = struct.unpack('>Q', size_binary)[0]
    _ = stream.read(size) # sort_id
    # size + proto_binary
    size_binary = stream.read(size_t)[::-1]
    size = struct.unpack('>Q', size_binary)[0]
    proto_binary = stream.read(size)
    instance = proto_parser_pb2.Instance()
    instance.ParseFromString(proto_binary)
    return instance


def parse_rec_instance(stream):
    size_t = 8
    # size + sort_id + size
    size_binary = stream.read(size_t)[::-1]
    size = struct.unpack('>Q', size_binary)[0]
    _ = stream.read(size) # sort_id
    stream.read(size_t)
    # size + proto_binary
    size_binary = stream.read(size_t)[::-1]
    size = struct.unpack('>Q', size_binary)[0]
    proto_binary = stream.read(size)
    instance = proto_parser_pb2.Instance()
    instance.ParseFromString(proto_binary)
    return instance


def parse_uint64_fid(value):
    binary_str = bin(np.uint64(value))
    return int(binary_str[:-54], 2), int(binary_str[-54:], 2)


def dump_tf_example(instance):
    slot_to_fid = {}
    for item in instance.fid:
        slot, fid = parse_uint64_fid(item)
        if slot in slot_to_fid:
            slot_to_fid[slot].append(fid)
        else:
            slot_to_fid[slot] = [fid]
    tmp_dict = {}
    for slot in range(1024):
        key = 'slot_%d' % slot
        fid_list = slot_to_fid.get(slot, [])
        fids_pb = feature_pb2.Int64List(value=fid_list)
        tmp_dict[key] = feature_pb2.Feature(int64_list=fids_pb)
    labels_pb = feature_pb2.FloatList(value=list(instance.label))
    tmp_dict['labels'] = feature_pb2.Feature(float_list=labels_pb)
    features_pb = feature_pb2.Features(feature=tmp_dict)
    return example_pb2.Example(features=features_pb)


def write_tf_record(example):
    data = example.SerializeToString()
    data_size = struct.pack('<Q', len(data))
    fake_data = ""
    fake_data_size = struct.pack('<Q', len(fake_data))
    res = "".join([data_size, data, fake_data_size, fake_data])
    print(res, end="")


def process_data(stream):
    count = 0
    while True:
        try:
            instance = parse_ad_instance(stream)
            example = dump_tf_example(instance)
            write_tf_record(example)
            count += 1
            logging.info(count)
        except Exception as e:
            logging.info("read failed: %s", e)
            break


if __name__ == "__main__":
    process_data(sys.stdin)

