# ============================================================
# scripts/test_model_load.py
# Condition C v3 local model-load smoke test
# ============================================================

import os
import tensorflow as tf

from keras.saving import register_keras_serializable


# ============================================================
# CUSTOM CBAM LAYERS
# ============================================================

@register_keras_serializable()
class ChannelAttention(tf.keras.layers.Layer):

    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):

        channels = int(input_shape[-1])

        hidden = max(
            channels // self.ratio,
            1
        )

        self.fc1 = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )

        self.fc2 = tf.keras.layers.Dense(
            channels
        )

        super().build(input_shape)

    def call(self, x):

        avg = tf.reduce_mean(
            x,
            axis=[1, 2],
            keepdims=True
        )

        mx = tf.reduce_max(
            x,
            axis=[1, 2],
            keepdims=True
        )

        attention = tf.nn.sigmoid(
            self.fc2(
                self.fc1(avg)
            )
            +
            self.fc2(
                self.fc1(mx)
            )
        )

        return x * attention

    def get_config(self):

        config = super().get_config()

        config.update({
            "ratio": self.ratio
        })

        return config


@register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.conv = tf.keras.layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid"
        )

        super().build(input_shape)

    def call(self, x):

        avg = tf.reduce_mean(
            x,
            axis=-1,
            keepdims=True
        )

        mx = tf.reduce_max(
            x,
            axis=-1,
            keepdims=True
        )

        concat = tf.concat(
            [avg, mx],
            axis=-1
        )

        return x * self.conv(
            concat
        )


# ============================================================
# PATH
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "condition_c_v3_best.keras"
)


# ============================================================
# CHECK
# ============================================================

print("=" * 70)
print("CONDITION C v3 — LOCAL MODEL LOAD TEST")
print("=" * 70)

print("Model:")
print(MODEL_PATH)

print()
print(
    "Exists:",
    os.path.exists(MODEL_PATH)
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        MODEL_PATH
    )


# ============================================================
# LOAD
# ============================================================

model = tf.keras.models.load_model(

    MODEL_PATH,

    custom_objects={
        "ChannelAttention":
            ChannelAttention,

        "SpatialAttention":
            SpatialAttention
    },

    compile=False,

    safe_mode=False
)


print()
print("MODEL LOADED SUCCESSFULLY")

print(
    "Input shape :",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)


# ============================================================
# VALIDATION
# ============================================================

assert model.input_shape[1:] == (
    15,
    224,
    224,
    3
)

assert model.output_shape[-1] == 2

print()
print("MODEL SHAPE CHECK PASSED")
print("=" * 70)