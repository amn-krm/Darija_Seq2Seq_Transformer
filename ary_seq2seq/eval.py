#!/usr/bin/env python3

from functools import partial
from pathlib import Path
from typing import Annotated

from keras.saving import load_model
from loguru import logger
import typer

from ary_seq2seq.modeling.ary_kh import TrainContext


# Simply subclass our TrainContext to keep it DRY
class InferenceContext(TrainContext):
	def __init__(self, model_path: Path) -> None:
		# Just fudge the exp_dir to get everything to look in the right place
		self.exp_dir = model_path
		model_file = self.exp_dir / "ary.keras"
		if not model_file.exists():
			model_file = self.exp_dir / "model.keras"

		self.load_trained_tokenizers()

		self.transformer = load_model(model_file.as_posix(), compile=True, safe_mode=True)
		# No inheritance, we very much don't care about what happens in TrainContext's ctor!


# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()


@app.command()
def main(
	model_path: Annotated[
		Path, typer.Option("--model-dir", "-d", help="Path to the folder containing the serialized model to evaluate")
	],
):
	logger.info("Loading model @ <magenta>model_path</magenta>")

	ctx = InferenceContext(model_path)

	ctx.load_dataset()
	ctx.clean_dataset()
	ctx.split_dataset()

	ctx.sample_inference(ctx.test_pairs, 100, "test_predictions.json")


if __name__ == "__main__":
	app()
