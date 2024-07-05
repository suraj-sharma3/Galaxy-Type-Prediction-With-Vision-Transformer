
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class ClassToken(Layer):
    def __init__(self):
        super().__init__()
    # trainable / learnable class embedding
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def mlp(x, cf):
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    return x

def transformer_encoder(x, cf):
    skip_1 = x # For skip connections
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
    )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, cf)
    x = Add()([x, skip_2])

    return x

def ViT(cf):
    """ Inputs """
    input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    inputs = Input(input_shape)     # (None, 64, 1875) (None - batch size, 64 - number of patches in an image and 1875 - patch width * patch height * patch channels)
    # print(inputs.shape)

    """ Patch Embeddings + Position Embeddings """
    patch_embed = Dense(cf["hidden_dim"])(inputs)   ## (None, 256, 768)
    # print(patch_embed.shape) # (None, 64, 768)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1) # delta is the step
    # print(positions)

    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions) ## (256, 768)
    # print(pos_embed.shape) ## (64, 768)
    embed = patch_embed + pos_embed 
    # print(embed.shape) ## (None, 64, 768)

    """ Adding Class Token """
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed]) ## (token - learnable class embedding & position embeddings of patches)
    # print(x.shape) # (None, 65, 768) (none - batch size, 65 - position embeddings for the 64 patches + 1 trainable / learnable class embedding, 768 - flattened patch)

    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    # print(x.shape) # output of the transformer encoder block # (None, 65, 768)

    """ Classification Head """
    x = LayerNormalization()(x)     ## (None, 257, 768)
    x = x[:, 0, :] 
    # print(x.shape) # (None, 768) # Input for the classification layer
    x = Dense(cf["num_classes"], activation="softmax")(x)
    print(x.shape) # (None, 10) # Output for the classification layer

    model = Model(inputs, x)
    return model


if __name__ == "__main__":
    config = {}
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["num_patches"] = 64
    config["patch_size"] = 25
    config["num_channels"] = 3
    config["num_classes"] = 10

    model = ViT(config)
    model.summary() # Prints the entire summary of the model in tabular form
