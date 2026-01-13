# Mangle Keras-Hub's TransformerDecoder to use FFNSwiGLU2...

import keras
from keras_hub.layers import TransformerDecoder

from ary_seq2seq.modeling.colmo import FFNSwiGLU2


@keras.saving.register_keras_serializable()
class TransformerDecoderSwiGLU(TransformerDecoder):
	# Overload the minimal amount of stuff possible:
	# we just want to overwrite the first FFN to avoid code duplication...
	# On the flipside, if the internal fields ever change names, we're screwed ;).
	def build(
		self,
		decoder_sequence_shape,
		encoder_sequence_shape=None,
		**kwargs,
	):
		# Whee, inheritance magic!
		super(TransformerDecoderSwiGLU, self).build(
			decoder_sequence_shape, encoder_sequence_shape=encoder_sequence_shape, **kwargs
		)

		# The segment of the graph we care about basically looks like:
		# `_feedforward_intermediate_dense` -> `_feedforward_output_dense` -> `_feedforward_dropout`
		# (with a `_feedforward_layer_norm` either at the start or at the end)
		# We only want to tweak the *first* FFN.

		# Replace Dense w/ FFNSwiGLU2
		self._feedforward_intermediate_dense = FFNSwiGLU2(
			self.intermediate_dim,
			dtype=self.dtype_policy,
			name="feedforward_intermediate_swiglu",
		)
		self._feedforward_intermediate_dense.build(decoder_sequence_shape)

		# Replace LayerNormalization w/ RMSNormalization
		self._feedforward_layer_norm = keras.layers.RMSNormalization(
			dtype=self.dtype_policy,
			name="feedforward_layer_rmsnorm",
		)
		self._feedforward_layer_norm.build(decoder_sequence_shape)
