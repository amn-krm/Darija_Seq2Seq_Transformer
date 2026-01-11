#!/usr/bin/env python3

import os
import random

# NOTE: Switch to the torch backend,
#       performance *may* be better on ROCm for some workflows...
#       Here, in practice, it appears to be faster for epoch 1,
#       but slower after that (and GPU/power utilization is not maximized),
#       so we end up using the tf backend.
# os.environ["KERAS_BACKEND"] = "torch"  # noqa: E402
# NOTE: And when we're *not* using the torch backend,
#       don't let tensorflow alloc a giant block of VRAM on startup,
#       because it makes for a *really* bad time when the driver decides
#       to relocate it to GTT to accomodate another smaller alloc...
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # noqa: E402
# NOTE: Same deal for XLA and its initial 75% of VRAM chunk...
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # noqa: E402
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

# Flash Attention should no longer be experimental on my GPU in ROCm...
keras.config.enable_flash_attention()  # noqa: E402
from keras import ops
import keras_hub
from loguru import logger
from matplotlib import pyplot as plt
import sentencepiece as spm
import tensorflow as tf

# import torch
# Raise errors ASAP
# torch.autograd.set_detect_anomaly(True)  # noqa: E402
import typer
from typing_extensions import Annotated

from ary_seq2seq.config import ATLASET_DATASET, REPORTS_DIR
from ary_seq2seq.modeling.layers import TransformerDecoderSwiGLU

type SentPair = tuple[str, str]
type SentPairList = list[SentPair]

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()

# -------- experiment directory --------
EXP_NAME = "darija_en_transformer_spm_kh"

# parms/hparms
BATCH_SIZE = 128
EPOCHS = 20
SEQUENCE_LENGTH = 50
VOCAB_SIZE = 30_000
START_TOKEN = "[start]"
END_TOKEN = "[end]"
PAD_ID = 0

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8

# Decoder-specific
# 8/3 * embedding with SwiGLU to keep n. of computations constant
FEED_FORWARD_DIM = int(EMBED_DIM * 8 / 3)

# -------- cleaning utilities --------
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REF_RE = re.compile(r"\[\d+\]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u061c]")


def clean_text(text: str) -> str:
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

	MAX_ROWS = 500_000
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

		pairs.append((en, darija))

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


# Torch-compatible Dataset (because tf.Dataset is pain.)
class TranslationDataset(keras.utils.PyDataset):
	def __init__(self, sp_en, sp_ary, pairs, **kwargs):
		super().__init__(**kwargs)

		self.sp_en = sp_en
		self.sp_ary = sp_ary
		self.eng, self.ary = zip(*pairs)

		# Pad/trim `enc` to `SEQUENCE_LENGTH`
		self.enc_start_end_packer = keras_hub.layers.StartEndPacker(
			sequence_length=SEQUENCE_LENGTH,
			pad_value=PAD_ID,
		)

		# Add special tokens (START & END) to `dec` and pad/trim it as well
		self.dec_start_end_packer = keras_hub.layers.StartEndPacker(
			sequence_length=SEQUENCE_LENGTH + 1,
			start_value=self.sp_ary.piece_to_id(START_TOKEN),
			end_value=self.sp_ary.piece_to_id(END_TOKEN),
			pad_value=PAD_ID,
		)

	def __len__(self):
		return math.ceil(len(self.eng) / BATCH_SIZE)

	def __getitem__(self, idx):
		start = idx * BATCH_SIZE
		end = start + BATCH_SIZE

		enc = self.enc_start_end_packer(tf.ragged.constant(self.sp_en.encode(list(self.eng[start:end]))))
		dec = self.dec_start_end_packer(tf.ragged.constant(self.sp_ary.encode(list(self.ary[start:end]))))

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

	# Cheap-ass way of handling our --with-swiglu flag w/o touching the function signature :D
	if EXP_NAME.endswith("_swiglu"):
		logger.info("Using custom <blue>TransformerDecoderSwiGLU</blue> layer!")
		x = TransformerDecoderSwiGLU(intermediate_dim=FEED_FORWARD_DIM, num_heads=NUM_HEADS)(
			decoder_sequence=x, encoder_sequence=encoded_seq_inputs
		)
	else:
		# NOTE: We get cross-attention by passing encoded_seq_inputs as encoder_sequence here
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

	transformer = keras.Model(
		[encoder_inputs, decoder_inputs],
		decoder_outputs,
		name="transformer",
	)

	return transformer


def train_model(
	exp_dir: Path, timestamp: str, transformer: keras.Model, train_ds: TranslationDataset, val_ds: TranslationDataset
) -> tuple[keras.Model, keras.callbacks.History]:
	logger.info("Training the model...")
	transformer.summary()

	# Setup our callbacks
	earlystop = keras.callbacks.EarlyStopping(
		monitor="val_loss",
		min_delta=0,
		patience=5,
		verbose=1,
		mode="min",
		baseline=None,
		restore_best_weights=True,
		start_from_epoch=8,  # We generally converge around epoch 15
	)

	tensorboard = keras.callbacks.TensorBoard(
		log_dir=(exp_dir / "logs").as_posix(),
		histogram_freq=1,
		write_graph=True,
		write_images=False,
		write_steps_per_second=False,
		update_freq="epoch",
		profile_batch=0,
		embeddings_freq=1,
		embeddings_metadata=None,
	)

	# In case we ever need to interrupt training...
	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath=(exp_dir / "checkpoints" / f"ary-{timestamp}.keras").as_posix(),
		monitor="val_loss",
		mode="min",
		save_best_only=True,
		save_freq="epoch",
		verbose=1,
	)

	transformer.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	history = transformer.fit(
		train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[earlystop, tensorboard, checkpoint], verbose=1
	)

	return transformer, history


def plot_training(exp_dir: Path, history: keras.callbacks.History) -> None:
	fig, ax = plt.subplots()
	ax.plot(history.history["loss"], label="Train loss")
	ax.plot(history.history["val_loss"], label="Val loss")
	ax.set_title("Loss")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss value")
	ax.legend()

	plt.savefig((exp_dir / "losses.png").as_posix())
	plt.clf()

	fig, ax = plt.subplots()
	ax.plot(history.history["accuracy"], label="Train accuracy")
	ax.plot(history.history["val_accuracy"], label="Val accuracy")
	ax.set_title("Accuracy")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy value")
	ax.legend()

	plt.savefig((exp_dir / "accuracy.png").as_posix())
	plt.clf()


# Inference
def decode_sequences(
	transformer: keras.Model,
	sp_en: spm.SentencePieceProcessor,
	sp_ary: spm.SentencePieceProcessor,
	input_sentences: list[str],
) -> list[str]:
	batch_size = 1

	# Tokenize the encoder input
	encoder_input_tokens = ops.convert_to_tensor(sp_en.encode(input_sentences, out_type=int), sparse=False, ragged=False)
	if ops.shape(encoder_input_tokens)[1] < SEQUENCE_LENGTH:
		pads = ops.zeros(
			(1, SEQUENCE_LENGTH - ops.shape(encoder_input_tokens)[1]),
			dtype=encoder_input_tokens.dtype,
		)
		encoder_input_tokens = ops.concatenate([encoder_input_tokens, pads], 1)
	elif ops.shape(encoder_input_tokens)[1] > SEQUENCE_LENGTH:
		encoder_input_tokens = encoder_input_tokens[..., :SEQUENCE_LENGTH]

	# Define a function that outputs the next token's probability given the input sequence
	def next(prompt: tf.Tensor, cache, index: int):
		logits = transformer([encoder_input_tokens, prompt])[:, index - 1, :]
		# Ignore hidden states for now; only needed for contrastive search
		hidden_states = None
		return logits, hidden_states, cache

	# Build a prompt of SEQUENCE_LENGTH with a start token and padding tokens
	length = SEQUENCE_LENGTH
	start = ops.full((batch_size, 1), sp_ary.piece_to_id(START_TOKEN))
	pad = ops.full((batch_size, length - 1), PAD_ID)
	prompt = ops.concatenate((start, pad), axis=-1)

	generated_tokens = keras_hub.samplers.GreedySampler()(
		next,
		prompt,
		stop_token_ids=[sp_ary.piece_to_id(END_TOKEN)],
		index=1,  # Start sampling after start token
	)
	# NOTE: spm doesn't deal with ndarrays all that well, cast to a list of Python ints...
	generated_sentences = sp_ary.decode(tf.squeeze(generated_tokens, 0).numpy().astype(int).tolist())
	# NOTE: spm automatically unwraps single-sentence outputs...
	if isinstance(generated_sentences, str):
		return [generated_sentences]
	else:
		return generated_sentences


def save_experiment(
	exp_dir: Path, transformer: keras.Model, sp_en: spm.SentencePieceProcessor, sp_ary: spm.SentencePieceProcessor
) -> Path:
	logger.info("Saving model...")

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
	exp_dir: Path,
	transformer: keras.Model,
	sp_en: spm.SentencePieceProcessor,
	sp_ary: spm.SentencePieceProcessor,
	test_pairs: SentPairList,
) -> None:
	NUM_EXAMPLES = 20
	examples = []

	for eng, ref in random.sample(test_pairs, NUM_EXAMPLES):
		pred = decode_sequences(transformer, sp_en, sp_ary, [eng])[0]
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
def main(with_swiglu: Annotated[bool, typer.Option(help="Use a Decoder w/ RMSNorm & a SwiGLU FFN")] = False):
	global EXP_NAME
	if with_swiglu:
		EXP_NAME += "_swiglu"

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

	timestamp = time.strftime("%Y%m%d_%H%M%S")
	exp_dir = REPORTS_DIR / f"{EXP_NAME}_{timestamp}"
	exp_dir.mkdir(parents=True, exist_ok=True)
	logger.info(f"Saving experiment run to <magenta>{exp_dir}</magenta>")

	transformer = build_model(eng_vocab_size, ary_vocab_size)
	transformer, history = train_model(exp_dir, timestamp, transformer, train_ds, val_ds)

	save_experiment(exp_dir, transformer, sp_en, sp_ary)
	plot_training(exp_dir, history)

	test_loss, test_acc = eval_on_test(transformer, test_ds)

	# Inference
	logger.info("Running an inference test on the trained model")
	for _ in range(5):
		s = random.choice(test_pairs)[0]
		print("ENG:", s)
		print("ARY:", decode_sequences(transformer, sp_en, sp_ary, [s])[0])
		print()

	sample_inference(exp_dir, transformer, sp_en, sp_ary, test_pairs)

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
