#!/usr/bin/env python3

# ============================================================
# 1. Backend setup
# ============================================================
import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
torch.autograd.set_detect_anomaly(True)

# ============================================================
# 2. Imports
# ============================================================
import math
import random
import re
import ast
import html
import unicodedata
import tempfile
import json
import time
from pathlib import Path

import numpy as np
import keras
from keras import layers, ops

from datasets import load_from_disk
import sentencepiece as spm

from ary_seq2seq.dataset import load_clean_dataset


# ============================================================
# 3. Cleaning utilities
# ============================================================
def standardize(text):
    return text.lower().strip()

DATASET_FRACTION = 1.0
pairs = load_clean_dataset(DATASET_FRACTION)

# ============================================================
# 5. Train / Val / Test split
# ============================================================
random.shuffle(pairs)

num_val = int(0.15 * len(pairs))
num_train = len(pairs) - 2 * num_val

train_pairs = pairs[:num_train]
val_pairs   = pairs[num_train:num_train + num_val]
test_pairs  = pairs[num_train + num_val:]

print(f"{len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test")

# ============================================================
# 6. Train SentencePiece tokenizers
# ============================================================
VOCAB_SIZE = 30_000

def train_spm(texts, prefix):
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        for t in texts:
            f.write(standardize(t) + "\n")
        fname = f.name

    spm.SentencePieceTrainer.train(
        input=fname,
        model_prefix=prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        user_defined_symbols=["[start]", "[end]"],
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
    )
    os.remove(fname)

print("Training EN tokenizer...")
train_spm([p[0] for p in train_pairs], "spm_en")

print("Training ARY tokenizer...")
train_spm([p[1] for p in train_pairs], "spm_ary")

# ============================================================
# 7. Load SentencePiece models
# ============================================================
sp_en = spm.SentencePieceProcessor()
sp_en.load("spm_en.model")

sp_ary = spm.SentencePieceProcessor()
sp_ary.load("spm_ary.model")

PAD_ID   = 0
START_ID = sp_ary.piece_to_id("[start]")
END_ID   = sp_ary.piece_to_id("[end]")

eng_vocab_size = sp_en.get_piece_size()
ary_vocab_size = sp_ary.get_piece_size()

print("ENG vocab:", eng_vocab_size)
print("ARY vocab:", ary_vocab_size)

# ============================================================
# 8. Vectorization utilities
# ============================================================
SEQUENCE_LENGTH = 50
BATCH_SIZE = 128

def pad_or_truncate(seq, max_len):
    seq = seq[:max_len]
    return seq + [PAD_ID] * (max_len - len(seq))

def encode_en(text):
    return pad_or_truncate(
        sp_en.encode(standardize(text), out_type=int),
        SEQUENCE_LENGTH
    )

def encode_ary(text):
    return pad_or_truncate(
        sp_ary.encode(standardize(text), out_type=int),
        SEQUENCE_LENGTH + 1
    )

# ============================================================
# 9. Torch-compatible Dataset
# ============================================================
class TranslationDataset(keras.utils.PyDataset):
    def __init__(self, pairs, **kwargs):
        super().__init__(**kwargs)

        self.eng, self.ary = zip(*pairs)

    def __len__(self):
        return math.ceil(len(self.eng) / BATCH_SIZE)

    def __getitem__(self, idx):
        start = idx * BATCH_SIZE
        end = start + BATCH_SIZE

        enc = np.array([encode_en(t) for t in self.eng[start:end]], dtype="int32")
        dec = np.array([encode_ary(t) for t in self.ary[start:end]], dtype="int32")

        return (
            {
                "encoder_inputs": enc,
                "decoder_inputs": dec[:, :-1],
            },
            dec[:, 1:],
        )

train_ds = TranslationDataset(train_pairs)
val_ds   = TranslationDataset(val_pairs)
test_ds  = TranslationDataset(test_pairs)

print(train_ds[0])
print(val_ds[0])
print(test_ds[0])


# ============================================================
# 8. Transformer layers
# ============================================================
from ary_seq2seq.modeling.torch_layers import TransformerEncoder, PositionalEmbedding, TransformerDecoder

# ============================================================
# 11. Build model
# ============================================================
EMBED_DIM = 256
LATENT_DIM = 2048
NUM_HEADS = 8

encoder_inputs = keras.Input((None,), dtype="int32", name="encoder_inputs")
x = PositionalEmbedding(SEQUENCE_LENGTH, eng_vocab_size, EMBED_DIM)(encoder_inputs)
encoder_outputs = TransformerEncoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x)

decoder_inputs = keras.Input((None,), dtype="int32", name="decoder_inputs")
x = PositionalEmbedding(SEQUENCE_LENGTH, ary_vocab_size, EMBED_DIM)(decoder_inputs)
x = TransformerDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)([x, encoder_outputs])
outputs = layers.Dense(ary_vocab_size)(x)

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    outputs,
)

transformer.compile(
    optimizer="rmsprop",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
    metrics=["accuracy"],
)

transformer.summary()

# ============================================================
# 12. Train
# ============================================================
transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# ============================================================
# 13. Inference
# ============================================================
def decode_sequence(sentence):
    enc = np.array([encode_en(sentence)], dtype="int32")
    decoded = [START_ID]

    for _ in range(SEQUENCE_LENGTH):
        dec = pad_or_truncate(decoded, SEQUENCE_LENGTH)
        dec = np.array([dec], dtype="int32")

        preds = transformer([enc, dec], training=False)

        next_id = int(ops.argmax(preds[0, len(decoded) - 1]))
        decoded.append(next_id)

        if next_id == END_ID:
            break

    pieces = [sp_ary.id_to_piece(i) for i in decoded]
    pieces = [p for p in pieces if p not in ("[start]", "[end]")]
    return sp_ary.decode_pieces(pieces)

# ============================================================
# 14. Evaluation
# ============================================================
test_loss, test_acc = transformer.evaluate(test_ds, verbose=0)
print("Test loss:", test_loss)
print("Test acc:", test_acc)

for _ in range(5):
    s = random.choice(test_pairs)[0]
    print("EN:", s)
    print("ARY:", decode_sequence(s))
    print()

# ============================================================
# 15. Save experiment artifacts
# ============================================================
import json
import time
from pathlib import Path

# -------- experiment directory --------
EXP_NAME = "darija_en_transformer_spm"
timestamp = time.strftime("%Y%m%d_%H%M%S")

exp_dir = Path("experiments") / f"{EXP_NAME}_{timestamp}"
exp_dir.mkdir(parents=True, exist_ok=True)

print("Saving experiment to:", exp_dir)

# ============================================================
# 15.1 Save model
# ============================================================
model_path = exp_dir / "model.keras"
transformer.save(model_path)

# ============================================================
# 15.2 Save SentencePiece vocabularies
# ============================================================
with open(exp_dir / "eng_vocab.txt", "w", encoding="utf-8") as f:
    for i in range(sp_en.get_piece_size()):
        f.write(sp_en.id_to_piece(i) + "\n")

with open(exp_dir / "ary_vocab.txt", "w", encoding="utf-8") as f:
    for i in range(sp_ary.get_piece_size()):
        f.write(sp_ary.id_to_piece(i) + "\n")

# ============================================================
# 15.3 Evaluate on test set
# ============================================================
test_ds = TranslationDataset(test_pairs)

test_loss, test_acc = transformer.evaluate(test_ds, verbose=0)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# ============================================================
# 15.4 Qualitative inference examples
# ============================================================
NUM_EXAMPLES = 20
examples = []

for eng, ref in random.sample(test_pairs, NUM_EXAMPLES):
    pred = decode_sequence(eng)
    examples.append(
        {
            "english": eng,
            "reference_darija": ref,
            "predicted_darija": pred,
        }
    )

with open(exp_dir / "inference_examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)
