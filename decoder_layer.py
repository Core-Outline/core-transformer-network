import tensorflow as tf
from tf.keras import Input, Model
from tf.keras.layers import Dense, Dropout, LayerNormalization

from attention_transformer import MultiHeadAttention


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    """
    Comprises:
    Masked MultiHeadAttention( with lookahead mask and padding mask)
    MultiHeadAttention( with padding mask)
    2 Dense layers
    Dropout layer
    """

    inputs = Input(shape=(None, d_model), name='inputs')
    enc_outputs = Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        name="attention_1"
    )(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })

    attention1 = LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        name="attention_2"
    )(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })

    attention2 = Dropout(rate=dropout)(attention2)
    attention2 = LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = Dense(units=units, activation='relu')(attention2)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )
