#!/usr/bin/env python3

import ast
from functools import partial
import html
import re
from typing import TypeAlias
import unicodedata

from datasets import DatasetDict, load_dataset
from loguru import logger
import polars as pl
from tqdm.rich import tqdm
import typer

from ary_seq2seq.config import ATLASET_DATASET, CLEAN_DATASET

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()

SentPair: TypeAlias = tuple[str, str]
SentPairList: TypeAlias = list[SentPair]
SentPairDicts: TypeAlias = list[dict[str, str]]

# Cleaning utilities
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REF_RE = re.compile(r"\[\d+\]")
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u061c]")

# Dataset pruning
DATA_MAX_WORDS = 50


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


def clean_dataset(dataset: DatasetDict) -> SentPairDicts:
	logger.info("Cleaning dataset...")
	pairs = []

	# NOTE: The test split is *very* tiny, so we just ignore it
	for ex in tqdm(dataset["train"]):
		try:
			meta = ast.literal_eval(ex["metadata"])
		except Exception as e:
			logger.warning(e)
			continue

		en = clean_text(meta.get("english", ""))
		darija = clean_text(ex["text"])

		if not en or not darija:
			continue
		if not (3 <= len(en.split()) <= DATA_MAX_WORDS):
			continue
		if not (3 <= len(darija.split()) <= DATA_MAX_WORDS):
			continue

		pairs.append({"eng": en, "ary": darija})

	logger.info(f"Total clean pairs: <green>{len(pairs)}</green>")
	return pairs


def load_clean_dataset(fraction: float) -> SentPairList:
	df = pl.read_parquet(CLEAN_DATASET)

	# NOTE: Previous behavior led to 321668 rows,
	#       we now have 349136 at most...
	return [tuple(d.values()) for d in df.sample(fraction=fraction).to_dicts()]


@app.command()
def main():
	logger.info("Downloading dataset from HF...")
	# NOTE: requires `hf auth login`
	ds = load_dataset("atlasia/Atlaset")

	logger.info("Dumping raw dataset to disk...")
	ds.save_to_disk(ATLASET_DATASET.as_posix())

	# Then clean the whole thing once and for all...
	pairs = clean_dataset(ds)

	logger.info("Dumping clean dataset to disk...")
	lf = pl.LazyFrame(pairs)
	lf.sink_parquet(CLEAN_DATASET)


if __name__ == "__main__":
	app()
