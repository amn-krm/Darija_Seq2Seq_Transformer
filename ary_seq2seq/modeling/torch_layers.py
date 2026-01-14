# Custom layers for the Keras/Torch variant

import torch
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = ops.cast(mask[:, None, :], "int32")

        attn = self.attention(inputs, inputs, attention_mask=mask)
        x = self.layernorm_1(inputs + attn)
        proj = self.dense_proj(x)
        return self.layernorm_2(x + proj)


@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(sequence_length, embed_dim)

    def call(self, x):
        length = ops.shape(x)[-1]
        positions = ops.arange(0, length)
        return self.token_emb(x) + self.pos_emb(positions)

    def compute_mask(self, x, mask=None):
        return ops.not_equal(x, 0)


@keras.saving.register_keras_serializable()
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True
        self.attn_1 = layers.MultiHeadAttention(num_heads, embed_dim)
        self.attn_2 = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.norm_1 = layers.LayerNormalization()
        self.norm_2 = layers.LayerNormalization()
        self.norm_3 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        x, enc = inputs

        if mask is not None:
            dec_mask, enc_mask = mask
        else:
            dec_mask, enc_mask = None, None

        causal = self.causal_mask(x)

        if dec_mask is not None:
            dec_mask = ops.cast(dec_mask[:, None, :], "int32")
            self_mask = ops.minimum(causal, dec_mask)
        else:
            self_mask = causal

        attn1 = self.attn_1(x, x, attention_mask=self_mask)
        x = self.norm_1(x + attn1)

        if enc_mask is not None:
            enc_mask = ops.cast(enc_mask[:, None, :], "int32")

        attn2 = self.attn_2(x, enc, attention_mask=enc_mask)
        x = self.norm_2(x + attn2)

        proj = self.dense_proj(x)
        return self.norm_3(x + proj)

    def causal_mask(self, x):
        n = ops.shape(x)[1]
        i = ops.arange(n)[:, None]
        j = ops.arange(n)
        mask = ops.cast(i >= j, "int32")
        return ops.expand_dims(mask, 0)
