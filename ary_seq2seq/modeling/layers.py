# Mangle Keras-Hub's TransformerDecoder to use FFNSwiGLU2...

import keras
from keras_hub.layers import TransformerDecoder
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.cached_multi_head_attention import (
	CachedMultiHeadAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer

from ary_seq2seq.modeling.colmo import FFNSwiGLU2


# NOTE: Because trying to do that in a slightly less brute-force approach w/ inheritance
#       led to some (potentially spurious) warnings (c.f., b4061753f1721cd0891cf96d04bf64f94929e7c0~),
#       here's a dumb variant that just copies over the whole method ;'(.
# From https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/layers/modeling/transformer_decoder.py @ c72a4aa
@keras_hub_export("keras_hub.layers.TransformerDecoderSwiGLU")
class TransformerDecoderSwiGLU(TransformerDecoder):
	# Copy and modify the full imp...
	def build(
		self,
		decoder_sequence_shape,
		encoder_sequence_shape=None,
	):
		self._decoder_sequence_shape = decoder_sequence_shape
		self._encoder_sequence_shape = encoder_sequence_shape
		# Infer the dimension of our hidden feature size from the build shape.
		hidden_dim = decoder_sequence_shape[-1]
		# Attention head size is `hidden_dim` over the number of heads.
		head_dim = int(hidden_dim // self.num_heads)
		if head_dim == 0:
			raise ValueError(
				"Attention `head_dim` computed cannot be zero. "
				f"The `hidden_dim` value of {hidden_dim} has to be equal to "
				f"or greater than `num_heads` value of {self.num_heads}."
			)

		# Self attention layers.
		self._self_attention_layer = CachedMultiHeadAttention(
			num_heads=self.num_heads,
			key_dim=head_dim,
			dropout=self.dropout,
			kernel_initializer=clone_initializer(self.kernel_initializer),
			bias_initializer=clone_initializer(self.bias_initializer),
			dtype=self.dtype_policy,
			name="self_attention",
		)
		if hasattr(self._self_attention_layer, "_build_from_signature"):
			self._self_attention_layer._build_from_signature(
				query=decoder_sequence_shape,
				value=decoder_sequence_shape,
			)
		else:
			self._self_attention_layer.build(
				query_shape=decoder_sequence_shape,
				value_shape=decoder_sequence_shape,
			)
		self._self_attention_layer_norm = keras.layers.LayerNormalization(
			epsilon=self.layer_norm_epsilon,
			dtype=self.dtype_policy,
			name="self_attention_layer_norm",
		)
		self._self_attention_layer_norm.build(decoder_sequence_shape)
		self._self_attention_dropout = keras.layers.Dropout(
			rate=self.dropout,
			dtype=self.dtype_policy,
			name="self_attention_dropout",
		)

		# Cross attention layers are optional.
		self._cross_attention_layer = None
		if encoder_sequence_shape:
			self._cross_attention_layer = CachedMultiHeadAttention(
				num_heads=self.num_heads,
				key_dim=head_dim,
				value_dim=head_dim,
				dropout=self.dropout,
				kernel_initializer=clone_initializer(self.kernel_initializer),
				bias_initializer=clone_initializer(self.bias_initializer),
				dtype=self.dtype_policy,
				name="cross_attention",
			)
			if hasattr(self._cross_attention_layer, "_build_from_signature"):
				self._cross_attention_layer._build_from_signature(
					query=decoder_sequence_shape,
					value=encoder_sequence_shape,
				)
			else:
				self._cross_attention_layer.build(
					query_shape=decoder_sequence_shape,
					value_shape=encoder_sequence_shape,
				)
			self._cross_attention_layer_norm = keras.layers.LayerNormalization(
				epsilon=self.layer_norm_epsilon,
				dtype=self.dtype_policy,
				name="cross_attention_layer_norm",
			)
			self._cross_attention_layer_norm.build(decoder_sequence_shape)
			self._cross_attention_dropout = keras.layers.Dropout(
				rate=self.dropout,
				dtype=self.dtype_policy,
				name="cross_attention_dropout",
			)

		# Feedforward layers.
		# NOTE: Replace Dense w/ FFNSwiGLU2
		self._feedforward_intermediate_dense = FFNSwiGLU2(
			self.intermediate_dim,
			dtype=self.dtype_policy,
			name="feedforward_intermediate_swiglu",
		)
		self._feedforward_intermediate_dense.build(decoder_sequence_shape)
		self._feedforward_output_dense = keras.layers.Dense(
			hidden_dim,
			kernel_initializer=clone_initializer(self.kernel_initializer),
			bias_initializer=clone_initializer(self.bias_initializer),
			dtype=self.dtype_policy,
			name="feedforward_output_dense",
		)
		intermediate_shape = list(decoder_sequence_shape)
		intermediate_shape[-1] = self.intermediate_dim
		self._feedforward_output_dense.build(tuple(intermediate_shape))
		# NOTE: Replace LayerNormalization w/ RMSNormalization
		self._feedforward_layer_norm = keras.layers.RMSNormalization(
			dtype=self.dtype_policy,
			name="feedforward_layer_rmsnorm",
		)
		self._feedforward_layer_norm.build(decoder_sequence_shape)
		self._feedforward_dropout = keras.layers.Dropout(
			rate=self.dropout,
			dtype=self.dtype_policy,
			name="feedforward_dropout",
		)
		# Create layers based on input shape.
		self.built = True
