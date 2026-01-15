#!/usr/bin/env python3

# Allow forward declarations of classes (in older Python versions)
from __future__ import annotations

import os
import random

# NOTE: Switch to the torch backend,
#       performance *may* be better on ROCm for some workflows...
#       Here, in practice, it appears to be faster for epoch 1,
#       but slower after that (and GPU/power utilization is not maximized),
#       so we end up using the tf backend.
#       (Tested inside AMD's tf 2.19 w/ rocm 7.1.1 docker container,
#       ... and an official pre-release torch wheel for rocm 7.1).
# os.environ["KERAS_BACKEND"] = "torch"  # noqa: E402
# NOTE: And when we're *not* using the torch backend,
#       don't let tensorflow alloc a giant block of VRAM on startup,
#       because it makes for a *really* bad time when the driver decides
#       to relocate it to GTT to accomodate another smaller alloc...
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # noqa: E402
# NOTE: Same deal for XLA and its initial 75% of VRAM chunk...
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # noqa: E402
random.seed(42)  # noqa: E402

import contextlib
from functools import partial
import json
import math
import time
from typing import Iterator

import keras

# Flash Attention should no longer be experimental on my GPU in ROCm...
keras.config.enable_flash_attention()  # noqa: E402
from keras import ops
import keras_hub
from loguru import logger
from matplotlib import pyplot as plt
import sentencepiece as spm
import tensorflow as tf
from tqdm.rich import tqdm

# import torch
# Raise errors ASAP
# torch.autograd.set_detect_anomaly(True)  # noqa: E402
import typer
from typing_extensions import Annotated

from ary_seq2seq.config import MODELS_DIR
from ary_seq2seq.dataset import SentPairList, load_clean_dataset
from ary_seq2seq.modeling.layers import TransformerDecoderSwiGLU

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()

# parms/hparms
DATASET_FRACTION = 1.0

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


class TrainContext:
	"""
	Wrap most of our stuff in a container class to be able to keep things modular while also keeping function signatures sane
	"""

	def __init__(self, with_swiglu: bool) -> None:
		self.exp_name = "darija_en_transformer_spm_kh"
		if with_swiglu:
			self.exp_name += "_swiglu"
		self.with_swiglu = with_swiglu

		self.timestamp = time.strftime("%Y%m%d_%H%M%S")
		self.exp_dir = MODELS_DIR / f"{self.exp_name}_{self.timestamp}"
		self.exp_dir.mkdir(parents=True, exist_ok=True)
		logger.info(f"Saving experiment run to <magenta>{self.exp_dir}</magenta>")

	def load_clean_dataset(self) -> None:
		logger.info("Loading clean dataset from disk...")
		self.pairs = load_clean_dataset(DATASET_FRACTION)

	def split_dataset(self) -> None:
		logger.info("Splitting dataset...")
		random.shuffle(self.pairs)

		num_val = int(0.15 * len(self.pairs))
		num_train = len(self.pairs) - 2 * num_val

		self.train_pairs = self.pairs[:num_train]
		self.val_pairs = self.pairs[num_train : num_train + num_val]
		self.test_pairs = self.pairs[num_train + num_val :]

		logger.info(
			f"<green>{len(self.train_pairs)}</green> train / <green>{len(self.val_pairs)}</green> val / <green>{len(self.test_pairs)}</green> test"
		)

	def train_tokenizers(self) -> None:
		with contextlib.chdir(self.exp_dir):
			logger.info("Training EN tokenizer...")
			train_spm((standardize(p[0]) for p in self.train_pairs), "spm_en")

			logger.info("Training ARY tokenizer...")
			train_spm((standardize(p[1]) for p in self.train_pairs), "spm_ary")

	# Load SentencePiece models
	def load_trained_tokenizers(self) -> None:
		self.sp_en = spm.SentencePieceProcessor()
		self.sp_en.load((self.exp_dir / "spm_en.model").as_posix())

		self.sp_ary = spm.SentencePieceProcessor()
		self.sp_ary.load((self.exp_dir / "spm_ary.model").as_posix())

		self.eng_vocab_size = self.sp_en.get_piece_size()
		self.ary_vocab_size = self.sp_ary.get_piece_size()

		logger.info(f"ENG vocab size: <green>{self.eng_vocab_size}</green>")
		logger.info(f"ARY vocab size: <green>{self.ary_vocab_size}</green>")

	def batch_dataset(self) -> None:
		self.train_ds = TranslationDataset(self.sp_en, self.sp_ary, self.train_pairs)
		self.val_ds = TranslationDataset(self.sp_en, self.sp_ary, self.val_pairs)
		self.test_ds = TranslationDataset(self.sp_en, self.sp_ary, self.test_pairs)

	def build_model(self) -> None:
		self.transformer = build_model(self.eng_vocab_size, self.ary_vocab_size, self.with_swiglu)

	def train_model(self) -> None:
		logger.info("Training the model...")
		self.transformer.summary()

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
			log_dir=(self.exp_dir / "logs").as_posix(),
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
			filepath=(self.exp_dir / "checkpoints" / f"ary-{self.timestamp}.keras").as_posix(),
			monitor="val_loss",
			mode="min",
			save_best_only=True,
			save_freq="epoch",
			verbose=1,
		)

		self.transformer.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

		self.train_hist = self.transformer.fit(
			self.train_ds,
			epochs=EPOCHS,
			validation_data=self.val_ds,
			callbacks=[earlystop, tensorboard, checkpoint],
			verbose=1,
		)

	def save_experiment(self) -> None:
		logger.info("Saving model...")

		# Save model
		model_path = self.exp_dir / "ary.keras"
		self.transformer.save(model_path)

		# Save vocabularies
		en_vocab = [self.sp_en.id_to_piece(i) for i in range(self.sp_en.get_piece_size())]
		(self.exp_dir / "eng_vocab.txt").write_text("\n".join(en_vocab), encoding="utf-8")

		ary_vocab = [self.sp_ary.id_to_piece(i) for i in range(self.sp_ary.get_piece_size())]
		(self.exp_dir / "ary_vocab.txt").write_text("\n".join(ary_vocab), encoding="utf-8")

		# Dump the history, too
		with (self.exp_dir / "history.json").open("w", encoding="utf-8") as f:
			json.dump(self.train_hist.history, f, ensure_ascii=False, indent=2)

	def plot_training(self) -> None:
		fig, ax = plt.subplots()
		ax.plot(self.train_hist.history["loss"], label="Train loss")
		ax.plot(self.train_hist.history["val_loss"], label="Val loss")
		ax.set_title("Loss")
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Loss value")
		ax.legend()

		plt.savefig((self.exp_dir / "losses.png").as_posix())
		plt.clf()

		fig, ax = plt.subplots()
		ax.plot(self.train_hist.history["accuracy"], label="Train accuracy")
		ax.plot(self.train_hist.history["val_accuracy"], label="Val accuracy")
		ax.set_title("Accuracy")
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Accuracy value")
		ax.legend()

		plt.savefig((self.exp_dir / "accuracy.png").as_posix())
		plt.clf()

	# Inference
	def decode_sequences(self, input_sentences: list[str]) -> list[str]:
		input_sentences = list(map(standardize, input_sentences))

		batch_size = 1

		# Tokenize the encoder input
		encoder_input_tokens = ops.convert_to_tensor(
			self.sp_en.encode(input_sentences, out_type=int), sparse=False, ragged=False
		)
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
			logits = self.transformer([encoder_input_tokens, prompt], training=False)[:, index - 1, :]
			# Ignore hidden states for now; only needed for contrastive search
			hidden_states = None
			return logits, hidden_states, cache

		# Build a prompt of SEQUENCE_LENGTH with a start token and padding tokens
		length = SEQUENCE_LENGTH
		start = ops.full((batch_size, 1), self.sp_ary.piece_to_id(START_TOKEN))
		pad = ops.full((batch_size, length - 1), PAD_ID)
		prompt = ops.concatenate((start, pad), axis=-1)

		generated_tokens = keras_hub.samplers.GreedySampler()(
			next,
			prompt,
			stop_token_ids=[self.sp_ary.piece_to_id(END_TOKEN)],
			index=1,  # Start sampling after start token
		)
		# NOTE: spm doesn't deal with ndarrays all that well, cast to a list of Python ints...
		generated_sentences = self.sp_ary.decode(tf.squeeze(generated_tokens, 0).numpy().astype(int).tolist())
		# NOTE: spm automatically unwraps single-sentence outputs...
		if isinstance(generated_sentences, str):
			return [generated_sentences]
		else:
			return generated_sentences

	# Evaluate on test set
	def eval_on_test(self) -> tuple[float, float]:
		logger.info("Evaluating model on the test set...")

		test_loss, test_acc = self.transformer.evaluate(self.test_ds, verbose=0)

		logger.info(f"Test loss: <blue>{test_loss:.4f}</blue>")
		logger.info(f"Test accuracy: <blue>{test_acc:.4f}</blue>")
		return test_loss, test_acc

	# Qualitative inference examples
	def sample_inference(
		self, test_pairs: SentPairList, num_examples: int = 20, filename: str = "inference_examples.json"
	) -> None:
		logger.info("Inferencing...")
		examples = []

		for eng, ref in tqdm(random.sample(test_pairs, num_examples)):
			pred = self.decode_sequences([eng])[0]
			examples.append(
				{
					"english": eng,
					"reference_darija": ref,
					"predicted_darija": pred,
				}
			)

		# Save examples
		with (self.exp_dir / filename).open("w", encoding="utf-8") as f:
			json.dump(examples, f, ensure_ascii=False, indent=2)


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

		enc = self.enc_start_end_packer(tf.ragged.constant(self.sp_en.encode(list(standardize(self.eng[start:end])))))
		dec = self.dec_start_end_packer(tf.ragged.constant(self.sp_ary.encode(list(standardize(self.ary[start:end])))))

		return (
			{
				"encoder_inputs": enc,
				"decoder_inputs": dec[:, :-1],
			},
			dec[:, 1:],
		)


# Build model
def build_model(eng_vocab_size: int, ary_vocab_size: int, with_swiglu: bool) -> keras.Model:
	logger.info("Building the model...")

	# Encoder
	encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

	x = keras_hub.layers.TokenAndPositionEmbedding(
		vocabulary_size=eng_vocab_size,
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
		vocabulary_size=ary_vocab_size,
		sequence_length=SEQUENCE_LENGTH,
		embedding_dim=EMBED_DIM,
	)(decoder_inputs)

	if with_swiglu:
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
	decoder_outputs = keras.layers.Dense(ary_vocab_size, activation="softmax")(x)
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


@app.command()
def main(with_swiglu: Annotated[bool, typer.Option(help="Use a Decoder w/ RMSNorm & a SwiGLU FFN")] = False):
	ctx = TrainContext(with_swiglu)

	ctx.load_clean_dataset()
	ctx.split_dataset()

	ctx.train_tokenizers()
	ctx.load_trained_tokenizers()

	ctx.batch_dataset()

	logger.info("Sanity-check of the data splits:")
	logger.info("train:")
	print(ctx.train_ds[0])
	logger.info("val:")
	print(ctx.val_ds[0])
	logger.info("test:")
	print(ctx.test_ds[0])

	ctx.build_model()
	ctx.train_model()

	ctx.save_experiment()
	ctx.plot_training()

	test_loss, test_acc = ctx.eval_on_test()

	# Inference
	logger.info("Running an inference test on the trained model")
	for _ in range(5):
		s = random.choice(ctx.test_pairs)[0]
		print("ENG:", s)
		print("ARY:", ctx.decode_sequences([s])[0])
		print()

	ctx.sample_inference(ctx.test_pairs)

	# Save training metadata
	experiment_metadata = {
		"experiment_name": ctx.exp_name,
		"timestamp": ctx.timestamp,
		"dataset": "Atlaset_corpus",
		"num_pairs_total": len(ctx.pairs),
		"num_train": len(ctx.train_pairs),
		"num_val": len(ctx.val_pairs),
		"num_test": len(ctx.test_pairs),
		"sequence_length": SEQUENCE_LENGTH,
		"batch_size": BATCH_SIZE,
		"embedding_dim": EMBED_DIM,
		"latent_dim": INTERMEDIATE_DIM,
		"swiglu_dim": FEED_FORWARD_DIM,
		"num_heads": NUM_HEADS,
		"optimizer": "RMSprop",
		"epochs": EPOCHS,
		"eng_vocab_size": ctx.eng_vocab_size,
		"ary_vocab_size": ctx.ary_vocab_size,
		"test_loss": float(test_loss),
		"test_accuracy": float(test_acc),
	}

	with (ctx.exp_dir / "experiments_metadata.json").open("w", encoding="utf-8") as f:
		json.dump(experiment_metadata, f, indent=2)

	logger.info("Experiment artifacts saved")


if __name__ == "__main__":
	app()
