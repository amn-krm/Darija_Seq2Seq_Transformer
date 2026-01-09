#!/usr/bin/env python3

import os
# NOTE: Switch to the torch backend, it appears to be slightly faster on AMD.
os.environ["KERAS_BACKEND"] = "torch"


import pathlib
import random

from datasets import load_from_disk, Dataset
import keras_hub
import keras
from keras import ops
import torch
# Raise errors ASAP
torch.autograd.set_detect_anomaly(True)

from ary_seq2seq.config import ATLASET_DATASET

# parms/hparms
BATCH_SIZE = 64
EPOCHS = 25
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 15000
ARY_VOCAB_SIZE = 15000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8


def load_dataset() -> Dataset:
	return load_from_disk(ATLASET_DATASET)

