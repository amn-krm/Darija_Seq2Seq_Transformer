#!/usr/bin/env python3

import re

from keras.saving import load_model
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, Label, Pretty

from ary_seq2seq.config import PRETRAINED_MODEL
from ary_seq2seq.modeling.ary_kh import END_TOKEN, START_TOKEN, TrainContext


# Simply subclass our TrainContext to keep it DRY
class InferenceContext(TrainContext):
	def __init__(self) -> None:
		# Just fudge the exp_dir to get everything to look in the right place
		self.exp_dir = PRETRAINED_MODEL
		model_file = self.exp_dir / "ary.keras"

		self.target_pattern = re.compile(rf"{re.escape(START_TOKEN)}(.*?){re.escape(END_TOKEN)}")

		self.load_trained_tokenizers()

		self.transformer = load_model(model_file.as_posix(), compile=True, safe_mode=True)
		# No inheritance, we very much don't care about what happens in TrainContext's ctor!

	def translate(self, sentence: str) -> str:
		# Thankfully, we already have basically all the scaffolding in place, so this is pretty simple
		prediction = self.decode_sequences([sentence])[0]

		m = self.target_pattern.fullmatch(prediction)
		if m:
			return m.group(1).strip()
		else:
			return prediction


# Very simple implementation based on the example InputApp...
class EnAry(App):
	CSS = """
    Input.-valid {
        border: tall $success 60%;
    }
    Input.-valid:focus {
        border: tall $success;
    }
    Input {
        margin: 1 1;
    }
    Label {
        margin: 1 2;
    }
    Pretty {
        margin: 1 2;
    }
"""

	# Make the ^q binding visible, otherwise you'd have to wait for Textual's popup on ^C
	BINDINGS = [
		Binding(key="^q", action="quit", description="Quit the app"),
	]

	def __init__(self) -> None:
		self.ary_ctx = InferenceContext()
		super().__init__()

	def compose(self) -> ComposeResult:
		yield Header()
		yield Footer()

		yield Label("Enter your source sentence (in English).")

		yield Input(
			placeholder="Would you like to play a game?",
			type="text",
			max_length=150,  # Eyeball estimation of what might be able to fit inside our context window
			id="prompt",
		)
		yield Pretty([], id="answer")

	@on(Input.Submitted)
	def translate(self, event: Input.Submitted) -> None:
		answer = self.query_one("#answer")
		translation = self.ary_ctx.translate(event.value)
		answer.update(f"{event.value} -> {translation}")

		prompt = self.query_one("#prompt")
		prompt.clear()

	def on_mount(self) -> None:
		self.title = "EN - ARY NMT"
		self.sub_title = "Prompt"


app = EnAry()

if __name__ == "__main__":
	app.run()
