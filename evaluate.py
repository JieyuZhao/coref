#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np

import tensorflow as tf
import util as util

def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = util.get_model(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  with tf.Session() as session:
    model.restore(session)
    trigger_ids = np.array([[int(x) for x in sys.argv[2:]]]) #[ 1109 13307 12680]
    # Make sure eval mode is True if you want official conll results
    model.evaluate(session, official_stdout=True, eval_mode=True, trigger_token_ids=trigger_ids)
