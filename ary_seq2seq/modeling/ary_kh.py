#!/usr/bin/env python3

import os
import random

# NOTE: Switch to the torch backend, it appears to be slightly faster on AMD.
os.environ["KERAS_BACKEND"] = "torch"  # noqa: E402
random.seed(42)  # noqa: E402

import ast
from functools import partial
import html
import json
import math
from pathlib import Path
import re
import time
from typing import Iterator
import unicodedata

from datasets import Dataset, load_from_disk
import keras
from keras import ops
import keras_hub
from loguru import logger
import numpy as np
import sentencepiece as spm
import torch
import typer

from ary_seq2seq.config import ATLASET_DATASET, REPORTS_DIR

# Raise errors ASAP
torch.autograd.set_detect_anomaly(True)

type SentPair = tuple[str, str]
type SentPairList = list[SentPair]

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()

# -------- experiment directory --------
EXP_NAME = "darija_en_transformer_spm"

# parms/hparms
BATCH_SIZE = 128
EPOCHS = 10
SEQUENCE_LENGTH = 50
VOCAB_SIZE = 30_000
START_TOKEN = "[start]"
END_TOKEN = "[end]"
PAD_ID = 0

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8


# -------- cleaning utilities --------
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
	text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("So"))
	text = re.sub(r"\s+", " ", text)
	return text.strip()


# -------- load dataset --------
def load_dataset() -> Dataset:
	logger.info("Loading dataset from disk...")
	return load_from_disk(ATLASET_DATASET)


def clean_dataset(ds: Dataset) -> SentPairList:
	logger.info("Cleaning dataset...")
	pairs = []

	MAX_ROWS = 20_000  # FIXME: 500_000
	MAX_WORDS = 50

	for ex in ds["train"].select(range(MAX_ROWS)):
		try:
			meta = ast.literal_eval(ex["metadata"])
		except Exception as e:
			logger.warning(e)
			continue

		en = clean_text(meta.get("english", ""))
		darija = clean_text(ex["text"])

		if not en or not darija:
			continue
		if not (3 <= len(en.split()) <= MAX_WORDS):
			continue
		if not (3 <= len(darija.split()) <= MAX_WORDS):
			continue

		pairs.append((en, f"{START_TOKEN} {darija} {END_TOKEN}"))

	logger.info(f"Total clean pairs: <green>{len(pairs)}</green>")

	return pairs


def split_dataset(pairs: SentPairList) -> tuple[SentPairList, SentPairList, SentPairList]:
	logger.info("Splitting dataset...")
	random.shuffle(pairs)

	num_val = int(0.15 * len(pairs))
	num_train = len(pairs) - 2 * num_val

	train_pairs = pairs[:num_train]
	val_pairs = pairs[num_train : num_train + num_val]
	test_pairs = pairs[num_train + num_val :]

	logger.info(
		f"<green>{len(train_pairs)}</green> train / <green>{len(val_pairs)}</green> val / <green>{len(test_pairs)}</green> test"
	)

	return train_pairs, val_pairs, test_pairs


# Text standardization
def standardize(text: str) -> str:
	return text.lower().strip()


# Train SentencePiece tokenizers
def train_spm(texts: Iterator[str], prefix: str):
	spm.SentencePieceTrainer.train(
		sentence_iterator=texts,
		model_prefix=prefix,
		vocab_size=VOCAB_SIZE,
		model_type="bpe",
		character_coverage=0.9995,
		byte_fallback=True,
		user_defined_symbols=[START_TOKEN, END_TOKEN],
		pad_id=0,
		unk_id=1,
		bos_id=-1,
		eos_id=-1,
	)


def train_tokenizers(train_pairs: SentPairList) -> None:
	logger.info("Training EN tokenizer...")
	train_spm((p[0] for p in train_pairs), "spm_en")

	logger.info("Training ARY tokenizer...")
	train_spm((p[1] for p in train_pairs), "spm_ary")


# Load SentencePiece models
def load_tokenizers() -> tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
	sp_en = spm.SentencePieceProcessor()
	sp_en.load("spm_en.model")

	sp_ary = spm.SentencePieceProcessor()
	sp_ary.load("spm_ary.model")

	return sp_en, sp_ary


# Vectorization utilities
def pad_or_truncate(seq, max_len: int):
	seq = seq[:max_len]
	return seq + [PAD_ID] * (max_len - len(seq))


def encode_en(sp_en: spm.SentencePieceProcessor, text: str):
	return pad_or_truncate(sp_en.encode(standardize(text), out_type=int), SEQUENCE_LENGTH)


def encode_ary(sp_ary: spm.SentencePieceProcessor, text: str):
	return pad_or_truncate(sp_ary.encode(standardize(text), out_type=int), SEQUENCE_LENGTH + 1)


# Torch-compatible Dataset
class TranslationDataset(keras.utils.PyDataset):
	def __init__(self, sp_en, sp_ary, pairs, **kwargs):
		self.eng, self.ary = zip(*pairs)
		self.sp_en = sp_en
		self.sp_ary = sp_ary
		super().__init__(**kwargs)

	def __len__(self):
		return math.ceil(len(self.eng) / BATCH_SIZE)

	def __getitem__(self, idx):
		start = idx * BATCH_SIZE
		end = start + BATCH_SIZE

		enc = np.array([encode_en(self.sp_en, t) for t in self.eng[start:end]], dtype="int32")
		dec = np.array([encode_ary(self.sp_ary, t) for t in self.ary[start:end]], dtype="int32")

		return (
			{
				"encoder_inputs": enc,
				"decoder_inputs": dec[:, :-1],
			},
			dec[:, 1:],
		)


# Build model
def build_model(ENG_VOCAB_SIZE: int, ARY_VOCAB_SIZE: int) -> keras.Model:
	logger.info("Building the model...")
	# Encoder
	encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

	# TODO: Switch to RoPE & SwiGLU
	x = keras_hub.layers.TokenAndPositionEmbedding(
		vocabulary_size=ENG_VOCAB_SIZE,
		sequence_length=SEQUENCE_LENGTH,
		embedding_dim=EMBED_DIM,
	)(encoder_inputs)

	encoder_outputs = keras_hub.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(
		inputs=x
	)
	# encoder = keras.Model(encoder_inputs, encoder_outputs)

	# Decoder
	decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")
	encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

	x = keras_hub.layers.TokenAndPositionEmbedding(
		vocabulary_size=ARY_VOCAB_SIZE,
		sequence_length=SEQUENCE_LENGTH,
		embedding_dim=EMBED_DIM,
	)(decoder_inputs)

	x = keras_hub.layers.TransformerDecoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(
		decoder_sequence=x, encoder_sequence=encoded_seq_inputs
	)
	x = keras.layers.Dropout(0.5)(x)
	decoder_outputs = keras.layers.Dense(ARY_VOCAB_SIZE, activation="softmax")(x)
	decoder = keras.Model(
		[
			decoder_inputs,
			encoded_seq_inputs,
		],
		decoder_outputs,
	)
	decoder_outputs = decoder([decoder_inputs, encoder_outputs])

	# TODO: Add a tb cb
	transformer = keras.Model(
		[encoder_inputs, decoder_inputs],
		decoder_outputs,
		name="transformer",
	)

	return transformer


def train_model(transformer: keras.Model, train_ds: TranslationDataset, val_ds: TranslationDataset) -> keras.Model:
	logger.info("Training the model...")
	transformer.summary()
	transformer.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

	return transformer


# Inference
def decode_sequence(
	transformer: keras.Model,
	sp_en: spm.SentencePieceProcessor,
	sp_ary: spm.SentencePieceProcessor,
	sentence: str,
) -> str:
	START_ID = sp_ary.piece_to_id(START_TOKEN)
	END_ID = sp_ary.piece_to_id(END_TOKEN)

	enc = np.array([encode_en(sp_en, sentence)], dtype="int32")
	decoded = [START_ID]

	for _ in range(SEQUENCE_LENGTH):
		dec = pad_or_truncate(decoded, SEQUENCE_LENGTH)
		dec = np.array([dec], dtype="int32")

		preds = transformer(
			{
				"encoder_inputs": enc,
				"decoder_inputs": dec,
			},
			training=False,
		)

		next_id = int(ops.argmax(preds[0, len(decoded) - 1]))
		decoded.append(next_id)

		if next_id == END_ID:
			break

	pieces = [sp_ary.id_to_piece(i) for i in decoded]
	pieces = [p for p in pieces if p not in (START_TOKEN, END_TOKEN)]
	return sp_ary.decode_pieces(pieces)


def save_experiment(
	transformer: keras.Model, sp_en: spm.SentencePieceProcessor, sp_ary: spm.SentencePieceProcessor, timestamp: str
) -> Path:
	logger.info("Saving model...")

	exp_dir = REPORTS_DIR / f"{EXP_NAME}_{timestamp}"
	exp_dir.mkdir(parents=True, exist_ok=True)

	logger.info(f"Saving experiment to <magenta>{exp_dir}</magenta>")

	# Save model
	model_path = exp_dir / "ary.keras"
	transformer.save(model_path)

	# Save vocabularies
	en_vocab = [sp_en.id_to_piece(i) for i in range(sp_en.get_piece_size())]
	(exp_dir / "eng_vocab.txt").write_text("\n".join(en_vocab), encoding="utf-8")

	ary_vocab = [sp_ary.id_to_piece(i) for i in range(sp_ary.get_piece_size())]
	(exp_dir / "ary_vocab.txt").write_text("\n".join(ary_vocab), encoding="utf-8")

	return exp_dir


# Evaluate on test set
def eval_on_test(transformer: keras.Model, test_ds: TranslationDataset) -> tuple[float, float]:
	logger.info("Evaluating model on the test set...")

	test_loss, test_acc = transformer.evaluate(test_ds, verbose=0)

	logger.info(f"Test loss: <blue>{test_loss:.4f}</blue>")
	logger.info(f"Test accuracy: <blue>{test_acc:.4f}</blue>")
	return test_loss, test_acc


# Qualitative inference examples
def sample_inference(
	transformer: keras.Model,
	sp_en: spm.SentencePieceProcessor,
	sp_ary: spm.SentencePieceProcessor,
	test_pairs: SentPairList,
	exp_dir: Path,
) -> None:
	NUM_EXAMPLES = 20
	examples = []

	for eng, ref in random.sample(test_pairs, NUM_EXAMPLES):
		pred = decode_sequence(transformer, sp_en, sp_ary, eng)
		examples.append(
			{
				"english": eng,
				"reference_darija": ref,
				"predicted_darija": pred,
			}
		)

	# Save examples
	with open(exp_dir / "inference_examples.json", "w", encoding="utf-8") as f:
		json.dump(examples, f, ensure_ascii=False, indent=2)


@app.command()
def main():
	ds = load_dataset()
	pairs = clean_dataset(ds)
	train_pairs, val_pairs, test_pairs = split_dataset(pairs)

	train_tokenizers(train_pairs)
	sp_en, sp_ary = load_tokenizers()

	eng_vocab_size = sp_en.get_piece_size()
	ary_vocab_size = sp_ary.get_piece_size()

	logger.info(f"ENG vocab size: <green>{eng_vocab_size}</green>")
	logger.info(f"ARY vocab size: <green>{ary_vocab_size}</green>")

	train_ds = TranslationDataset(sp_en, sp_ary, train_pairs)
	val_ds = TranslationDataset(sp_en, sp_ary, val_pairs)
	test_ds = TranslationDataset(sp_en, sp_ary, test_pairs)

	logger.info("Sanity-check the data splits:")
	logger.info("train:")
	print(train_ds[0])
	logger.info("val:")
	print(val_ds[0])
	logger.info("test:")
	print(test_ds[0])

	transformer = build_model(eng_vocab_size, ary_vocab_size)
	transformer = train_model(transformer, train_ds, val_ds)

	timestamp = time.strftime("%Y%m%d_%H%M%S")
	exp_dir = save_experiment(transformer, sp_en, sp_ary, timestamp)

	test_loss, test_acc = eval_on_test(transformer, test_ds)

	# Inference
	logger.info("Running an inference test on the trained model")
	for _ in range(5):
		s = random.choice(test_pairs)[0]
		print("ENG:", s)
		print("ARY:", decode_sequence(transformer, sp_en, sp_ary, s))
		print()

	sample_inference(transformer, sp_en, sp_ary, test_pairs, exp_dir)

	# Save training metadata
	experiment_metadata = {
		"experiment_name": EXP_NAME,
		"timestamp": timestamp,
		"dataset": "Atlaset_corpus",
		"num_pairs_total": len(pairs),
		"num_train": len(train_pairs),
		"num_val": len(val_pairs),
		"num_test": len(test_pairs),
		"sequence_length": SEQUENCE_LENGTH,
		"batch_size": BATCH_SIZE,
		"embedding_dim": EMBED_DIM,
		"latent_dim": INTERMEDIATE_DIM,
		"num_heads": NUM_HEADS,
		"optimizer": "RMSprop",
		"epochs": 10,
		"eng_vocab_size": eng_vocab_size,
		"ary_vocab_size": ary_vocab_size,
		"test_loss": float(test_loss),
		"test_accuracy": float(test_acc),
	}

	with open(exp_dir / "experiments_metadata.json", "w", encoding="utf-8") as f:
		json.dump(experiment_metadata, f, indent=2)

	logger.info("Experiment artifacts saved")


if __name__ == "__main__":
	app()
