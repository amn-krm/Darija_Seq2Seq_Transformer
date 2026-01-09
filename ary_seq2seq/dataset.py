#!/usr/bin/env python3


from datasets import load_dataset
from loguru import logger
import typer

from ary_seq2seq.config import ATLASET_DATASET

app = typer.Typer()


@app.command()
def main():
	logger.info("Downloading dataset for HF...")
	# NOTE: requires `hf auth login`
	ds = load_dataset("atlasia/Atlaset")

	logger.info("Dumping dataset to disk...")
	ds.save_to_disk(ATLASET_DATASET.as_posix())


if __name__ == "__main__":
	app()
