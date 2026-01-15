#!/usr/bin/env python3

from functools import partial
import json
from pathlib import Path
from typing import Annotated

from loguru import logger
from sacrebleu.metrics import BLEU, CHRF
import typer

# Always enable color in loguru
logger = logger.opt(colors=True)
logger.opt = partial(logger.opt, colors=True)

app = typer.Typer()


def print_scores(scoring_file: Path) -> None:
	with scoring_file.open(encoding="utf-8") as f:
		data = json.load(f)

	# Check the first item to sniff the field names
	sample = data[0]
	pred_field = "prediction" in sample and "prediction" or "predicted_darija"
	ref_field = "reference" in sample and "reference" or "reference_darija"

	hyps = [x.get(pred_field).strip() for x in data]
	refs = [[x.get(ref_field).strip() for x in data]]

	bleu = BLEU()  # BLEU sacreBLEU (corpus)
	chrf = CHRF(word_order=2)  # chrF++

	logger.info(f"BLEU: <blue>{bleu.corpus_score(hyps, refs)}</blue>")
	logger.info(f"chrF: <cyan>{chrf.corpus_score(hyps, refs)}</cyan>")


@app.command()
def main(scoring_file: Annotated[Path, typer.Argument(help="Path to the JSON file to score")]):
	print_scores(scoring_file)


if __name__ == "__main__":
	app()
