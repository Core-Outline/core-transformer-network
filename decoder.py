import tensorflow as tf
from tf.keras import Input, Model
from tf.keras.layers import Embedding, Dropout
from pos_encoding import PositionalEncoding
from decoder_layer import decoder_layer

def decoder(
    vocab_size,
    num_layers,
    units,
    d_model,
    num_heads,
    dropout,
    name='decoder'
):
    """
    Comprises:
    Output embedding
    Positional Encoding (for memory of relative position)
    N Decoder layers
    """
    inputs = Input(shape=(None,), name='inputs')
    enc_outputs = Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = Input(shape=(1, 1, None), name='padding_mask')

    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f'decoder_layer_{i}'
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )

