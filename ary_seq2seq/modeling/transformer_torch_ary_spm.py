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

from ary_seq2seq.config import ATLASET_DATASET

# ============================================================
# 3. Cleaning utilities
# ============================================================
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REF_RE = re.compile(r"\[\d+\]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u061c]")

def clean_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = REF_RE.sub("", text)
    text = BIDI_RE.sub("", text)
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("So")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def standardize(text):
    return text.lower().strip()

# ============================================================
# 4. Load Atlaset dataset
# ============================================================
ds = load_from_disk(ATLASET_DATASET)

pairs = []
MAX_ROWS = 500_000
MAX_WORDS = 50

for ex in ds["train"].select(range(MAX_ROWS)):
    try:
        meta = ast.literal_eval(ex["metadata"])
    except Exception:
        continue

    en = clean_text(meta.get("english", ""))
    darija = clean_text(ex["text"])

    if not en or not darija:
        continue
    if not (3 <= len(en.split()) <= MAX_WORDS):
        continue
    if not (3 <= len(darija.split()) <= MAX_WORDS):
        continue

    pairs.append((en, f"[start] {darija} [end]"))

print("Total clean pairs:", len(pairs))
print(pairs[:5])

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
BATCH_SIZE = 64

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
    def __init__(self, pairs):
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
@keras.saving.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = ops.cast(mask[:, None, :], "int32")

        attn = self.attention(inputs, inputs, attention_mask=mask)
        x = self.layernorm_1(inputs + attn)
        proj = self.dense_proj(x)
        return self.layernorm_2(x + proj)


@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(sequence_length, embed_dim)

    def call(self, x):
        length = ops.shape(x)[-1]
        positions = ops.arange(0, length)
        return self.token_emb(x) + self.pos_emb(positions)

    def compute_mask(self, x, mask=None):
        return ops.not_equal(x, 0)


@keras.saving.register_keras_serializable()
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super().__init__()
        self.supports_masking = True
        self.attn_1 = layers.MultiHeadAttention(num_heads, embed_dim)
        self.attn_2 = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.norm_1 = layers.LayerNormalization()
        self.norm_2 = layers.LayerNormalization()
        self.norm_3 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        x, enc = inputs

        if mask is not None:
            dec_mask, enc_mask = mask
        else:
            dec_mask, enc_mask = None, None

        causal = self.causal_mask(x)

        if dec_mask is not None:
            dec_mask = ops.cast(dec_mask[:, None, :], "int32")
            self_mask = ops.minimum(causal, dec_mask)
        else:
            self_mask = causal

        attn1 = self.attn_1(x, x, attention_mask=self_mask)
        x = self.norm_1(x + attn1)

        if enc_mask is not None:
            enc_mask = ops.cast(enc_mask[:, None, :], "int32")

        attn2 = self.attn_2(x, enc, attention_mask=enc_mask)
        x = self.norm_2(x + attn2)

        proj = self.dense_proj(x)
        return self.norm_3(x + proj)

    def causal_mask(self, x):
        n = ops.shape(x)[1]
        i = ops.arange(n)[:, None]
        j = ops.arange(n)
        mask = ops.cast(i >= j, "int32")
        return ops.expand_dims(mask, 0)

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
    {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
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
transformer.fit(train_ds, epochs=10, validation_data=val_ds)

# ============================================================
# 13. Inference
# ============================================================
def decode_sequence(sentence):
    enc = np.array([encode_en(sentence)], dtype="int32")
    decoded = [START_ID]

    for _ in range(SEQUENCE_LENGTH):
        dec = pad_or_truncate(decoded, SEQUENCE_LENGTH)
        dec = np.array([dec], dtype="int32")

        preds = transformer(
            {"encoder_inputs": enc, "decoder_inputs": dec},
            training=False,
        )

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
