#!/usr/bin/env python3

import os
# NOTE: Switch to the torch backend, it appears to be slightly faster on AMD.
os.environ["KERAS_BACKEND"] = "torch"

from functools import partial
import math
import json
import time
from pathlib import Path
import random
from typing import TypeAlias
random.seed(42)

from loguru import logger
from datasets import load_from_disk, Dataset
import keras_hub
import keras
from keras.layers import StringLookup
from keras import ops
import numpy as np
import torch
import typer
# Raise errors ASAP
torch.autograd.set_detect_anomaly(True)

from ary_seq2seq.config import ATLASET_DATASET, REPORTS_DIR, MODELS_DIR

type SentPair = tuple[str, str]
type SentPairList = list[SentPair]

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()

# -------- experiment directory --------
EXP_NAME = "darija_en_transformer_baseline"

# parms/hparms
BATCH_SIZE = 64
EPOCHS = 25
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 15000
ARY_VOCAB_SIZE = 15000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8


# -------- cleaning utilities --------
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REF_RE = re.compile(r"\[\d+\]")
BIDI_RE = re.compile(
	r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u061c]"
)


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


# TODO: Fancier tokenizers?
# -------- load dataset --------
def load_dataset() -> Dataset:
	logger.info("Loading dataset from disk...")
	return load_from_disk(ATLASET_DATASET)


def clean_dataset(Dataset) -> SentPairList:
	logger.info("Cleaning dataset...")
	pairs = []

	MAX_ROWS = 20_000
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

	pairs.append((en, "[start] " + darija + " [end]"))

	logger.info(f"Total clean pairs: <green>{len(pairs)}</green>")

	return pairs


def split_dataset(pairs: SentPairList) -> tuple[SentPairList, SentPairList, SentPairList]:
	logger.info("Splitting dataset...")
	random.shuffle(pairs)

	num_val = int(0.15 * len(pairs))
	num_train = len(pairs) - 2 * num_val

	train_pairs = pairs[:num_train]
	val_pairs = pairs[num_train:num_train + num_val]
	test_pairs = pairs[num_train + num_val:]

	logger.info(f"<green>{len(train_pairs)}</green> train / <green>{len(val_pairs)}</green> val / <green>{len(test_pairs)}</green> test")


# Text standardization (pure Python)
def standardize(text):
	return text.lower()


# Vocabulary via StringLookup (Torch compatible)
def build_vocab(train_pairs: SentPairList) -> tuple[StringLookup, StringLookup]:
	logger.info("Building the vocabulary...")
	vocab_size = 30_000
	sequence_length = 50
	batch_size = 64

	def get_tokens(texts):
		for t in texts:
			for token in standardize(t).split():
				yield token

	eng_lookup = StringLookup(
		max_tokens=vocab_size,
		output_mode="int"
	)

	ary_lookup = StringLookup(
		max_tokens=vocab_size,
		output_mode="int"
	)

	eng_lookup.adapt(list(get_tokens([p[0] for p in train_pairs])))
	ary_lookup.adapt(list(get_tokens([p[1] for p in train_pairs])))

	logger.info(f"ENG vocab size: <green>{eng_lookup.vocabulary_size()}</green>")
	logger.info(f"ARY vocab size: <green>{ary_lookup.vocabulary_size()}</green>")

	return eng_lookup, ary_lookup


# Vectorization helpers
def vectorize_eng(texts: list[str], eng_lookup: StringLookup) -> list[np.array]:
	outputs = []
	for t in texts:
		tokens = standardize(t).split()[:sequence_length]
		token_ids = eng_lookup(tokens)
		outputs.append(ops.convert_to_numpy(token_ids))
	return outputs


def vectorize_ary(texts: list[str], ary_lookup: StringLookup) -> list[np.array]:
	outputs = []
	for t in texts:
		tokens = standardize(t).split()[: sequence_length + 1]
		token_ids = ary_lookup(tokens)
		outputs.append(ops.convert_to_numpy(token_ids))
	return outputs


# Torch-compatible dataset
def pad_sequences(seqs, max_len) -> list[np.array]:
	padded = np.zeros((len(seqs), max_len), dtype="int32")
	for i, s in enumerate(seqs):
		padded[i, :len(s)] = s
	return padded


class TranslationDataset(keras.utils.PyDataset):
	def __init__(self, pairs, eng_lookup=eng_lookup, ary_lookup=ary_lookup):
		self.eng, self.darija = zip(*pairs)
		self.eng_lookup = eng_lookup
		self.ary_lookup = ary_lookup

	def __len__(self):
		return math.ceil(len(self.eng) / batch_size)

	def __getitem__(self, idx):
		start = idx * batch_size
		end = start + batch_size

		eng = vectorize_eng(self.eng[start:end], self.eng_lookup)
		dar = vectorize_ary(self.darija[start:end], self.ary_lookup)

		eng = pad_sequences(eng, sequence_length)
		dar = pad_sequences(dar, sequence_length + 1)

		return (
			{
				"encoder_inputs": eng,
				"decoder_inputs": dar[:, :-1],
			},
			dar[:, 1:],
		)


# Build model
def build_model(ENG_VOCAB_SIZE: int, ARY_VOCAB_SIZE: int) -> keras.Model:
	logger.info("Building the model...")
	# Encoder
	encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

	# TODO: Switch to ROPE & GeLu
	x = keras_hub.layers.TokenAndPositionEmbedding(
		vocabulary_size=ENG_VOCAB_SIZE,
		sequence_length=MAX_SEQUENCE_LENGTH,
		embedding_dim=EMBED_DIM,
	)(encoder_inputs)

	encoder_outputs = keras_hub.layers.TransformerEncoder(
		intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
	)(inputs=x)
	encoder = keras.Model(encoder_inputs, encoder_outputs)

	# Decoder
	decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")
	encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

	x = keras_hub.layers.TokenAndPositionEmbedding(
		vocabulary_size=ARY_VOCAB_SIZE,
		sequence_length=MAX_SEQUENCE_LENGTH,
		embedding_dim=EMBED_DIM,
	)(decoder_inputs)

	x = keras_hub.layers.TransformerDecoder(
		intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
	)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
	x = keras.layers.Dropout(0.5)(x)
	decoder_outputs = keras.layers.Dense(SPA_VOCAB_SIZE, activation="softmax")(x)
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
	transformer.compile(
		"rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
	)
	transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

	return transformer


# Inference
def decode_sequence(keras.Model: transformer, sentence: str):
	enc = pad_sequences(vectorize_eng([sentence]), sequence_length)

	decoded_ids = [int(ary_lookup("[start]"))]
	end_id = int(ary_lookup("[end]"))

	for _ in range(sequence_length):
		dec = pad_sequences([decoded_ids], sequence_length)

		preds = transformer(
			{
				"encoder_inputs": enc,
				"decoder_inputs": dec,
			}
		)

		next_id = int(ops.argmax(preds[0, len(decoded_ids) - 1]))
		decoded_ids.append(next_id)

		if next_id == end_id:
			break

	return " ".join(ary_index_lookup[i] for i in decoded_ids)


def save_experiment(transformer: keras.Model):
	logger.info("Saving model...")
	timestamp = time.strftime("%Y%m%d_%H%M%S")

	exp_dir = REPORTS_DIR / f"{EXP_NAME}_{timestamp}"
	exp_dir.mkdir(parents=True, exist_ok=True)

	logger.info(f"Saving experiment to <magenta>{exp_dir}</magenta>")

	# Save model
	model_path = MODELS_DIR / "ary.keras"
	transformer.save(model_path)

	# Save vocabularies
	with open(REPORTS_DIR / "eng_vocab.txt", "w", encoding="utf-8") as f:
		for tok in eng_lookup.get_vocabulary():
			f.write(tok + "\n")

	with open(REPORTS_DIR / "ary_vocab.txt", "w", encoding="utf-8") as f:
		for tok in ary_lookup.get_vocabulary():
			f.write(tok + "\n")


# Evaluate on test set
def eval_on_test(transformer: keras.Model, test_pairs: SentPairList):
	logger.info("Evaluating model on the test set...")
	test_ds = TranslationDataset(test_pairs)

	test_loss, test_acc = transformer.evaluate(test_ds, verbose=0)

	print(f"Test loss: <blue>{test_loss:.4f}</blue>")
	print(f"Test accuracy: <blue>{test_acc:.4f}</blue>")


# Qualitative inference examples
def sample_inference(transformer: keras.Model, test_pairs: SentPairList):
	NUM_EXAMPLES = 20
	examples = []

	for eng, ref in random.sample(test_pairs, NUM_EXAMPLES):
		pred = decode_sequence(transformer, eng)
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


# Save training metadata (for paper)
def save_metadata():
	experiment_metadata = {
		"experiment_name": EXP_NAME,
		"timestamp": timestamp,
		"dataset": "Atlaset_corpus",
		"num_pairs_total": len(pairs),
		"num_train": len(train_pairs),
		"num_val": len(val_pairs),
		"num_test": len(test_pairs),
		"sequence_length": sequence_length,
		"batch_size": batch_size,
		"embedding_dim": embed_dim,
		"latent_dim": latent_dim,
		"num_heads": num_heads,
		"optimizer": "RMSprop",
		"epochs": 10,
		"eng_vocab_size": eng_vocab_size,
		"ary_vocab_size": ary_vocab_size,
		"test_loss": float(test_loss),
		"test_accuracy": float(test_acc),
	}

	with open(REPORTS_DIR / "experiment_metadata.json", "w", encoding="utf-8") as f:
		json.dump(experiment_metadata, f, indent=2)

	logger.info("Experiment artifacts saved")


@app.command()
def main():
	ds = load_dataset()
	pairs = clean_dataset(ds)
	train_pairs, val_pairs, test_pairs = split_dataset(pairs)

	eng_lookup, ary_lookup = build_vocab(train_pairs)

	train_ds = TranslationDataset(train_pairs, eng_lookup=eng_lookup, ary_lookup=ary_lookup)
	val_ds = TranslationDataset(val_pairs, eng_lookup=eng_lookup, ary_lookup=ary_lookup)

	logger.info("Sanity-check the first train sample:")
	x, y = train_ds[0]
	print(x["encoder_inputs"].shape)
	print(x["decoder_inputs"].shape)
	print(y.shape)

	transformer = build_model(eng_lookup.vocabulary_size(), ary_lookup.vocabulary_size())
	transformer = train_model(transformer)

	save_experiment(transformer)

	eval_on_test(transformer, test_pairs)

	# Inference
	logger.info("Running an inference test on the trained model")
	ary_vocab = ary_lookup.get_vocabulary()
	ary_index_lookup = dict(enumerate(ary_vocab))

	for _ in range(5):
		s = random.choice(test_pairs)[0]
		print("ENG:", s)
		print("ARY:", decode_sequence(transformer, s))
		print()

	sample_inference(transformer, test_pairs)

	save_metadata()


if __name__ == "__main__":
	app()
