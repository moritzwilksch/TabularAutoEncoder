import tensorflow as tf
from tensorflow import keras


class TabularAE(keras.Model):
    def __init__(
        self, embedding_cols: list, continuous_cols: list, cardinalities: dict
    ):
        super().__init__()
        self.embedding_cols = embedding_cols
        self.continuous_cols = continuous_cols
        self.cardinalities = cardinalities

        # ----------------- Categorical Embeddings -----------------
        self.embeddings = {}
        for col in self.embedding_cols:
            self.embeddings[col] = keras.layers.Embedding(
                input_dim=self.cardinalities[col] + 1, output_dim=10
            )

        # concatenation
        self.emb_concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

        # ----------------- Continuous Inputs -----------------
        self.cont_concat = tf.keras.layers.Concatenate()
        self.reshape_cont = tf.keras.layers.Reshape((-1,))
        self.all_concatenated = tf.keras.layers.Concatenate()

        # ----------------- Bottleneck -----------------
        self.dense1 = tf.keras.layers.Dense(units=64, activation="relu")
        self.bottleneck = tf.keras.layers.Dense(units=5, activation="linear")

        # ----------------- Outputs -----------------
        # continuous
        self.output_continuous = {
            col: tf.keras.layers.Dense(
                units=1, activation="linear", name=f"output_{col}"
            )
            for col in self.continuous_cols
        }

        # categorical
        self.output_categorical = {
            col: tf.keras.layers.Dense(
                units=self.cardinalities[col],
                activation="softmax",
                name=f"output_{col}",
            )
            for col in self.embedding_cols
        }

        self.outputs = self.output_categorical | self.output_continuous

    def call(self, inputs, return_bottleneck=False):
        embeddings = {
            col: self.embeddings[col](inputs[col]) for col in self.embedding_cols
        }
        embeddings = self.flatten(self.emb_concat(list(embeddings.values())))

        continuous_concatenated = self.cont_concat(
            [self.reshape_cont(inputs[col]) for col in self.continuous_cols]
        )

        all_concatenated = self.all_concatenated([embeddings, continuous_concatenated])

        x = self.dense1(all_concatenated)
        bottleneck = self.bottleneck(x)

        if return_bottleneck:
            return bottleneck

        outputs = {
            colname: layer(bottleneck) for colname, layer in self.outputs.items()
        }

        return outputs

    def decode(self, bottleneck):
        outputs = {
            colname: layer(bottleneck) for colname, layer in self.outputs.items()
        }

        return outputs
