#%%
from black import out
import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
import tensorflow as tf
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from tensorflow import keras

# from rich import print

#%%
class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset = dataset_name

    def _load_raw(self):
        return sns.load_dataset(self.dataset)

    def _process(self, data: pd.DataFrame):
        embedding_cols = ["sex", "smoker", "day", "time"]
        continuous_cols = ["total_bill", "size"]

        # cat to integer
        self.oe = OrdinalEncoder(dtype=np.int8)
        data[embedding_cols] = self.oe.fit_transform(data[embedding_cols])

        # scale continuous
        self.scaler = StandardScaler()
        data[continuous_cols] = self.scaler.fit_transform(data[continuous_cols])

        return data

    def load(self):
        raw = self._load_raw()
        processed = self._process(raw)
        return processed


#%%
df = DataLoader("tips").load()
embedding_cols = ["sex", "smoker", "day", "time"]
continuous_cols = ["total_bill", "size"]

#%%
target_data = {col: pd.get_dummies(df[col].astype("category")) for col in embedding_cols}
target_data["output_cont"] = df[continuous_cols]

dataset = tf.data.Dataset.from_tensor_slices(
    (
        {col: df[col] for col in embedding_cols + continuous_cols},
        target_data,
    )
)

target_data = tf.data.Dataset.from_tensor_slices(
    target_data
)

print(target_data)
#%%
class NNModel:
    def __init__(self, embedding_cols: list, continuous_cols: list):
        self.embedding_cols = embedding_cols
        self.continuous_cols = continuous_cols

    def _build_models(self):
        # ---------- inputs ----------
        inputs = {}
        for col in self.embedding_cols + self.continuous_cols:
            inputs[col] = keras.Input(shape=(1,), name=col)

        # ---------- embedding layers ----------
        embeddings = {}
        for col in self.embedding_cols:
            cardinality = df[col].nunique()
            print(f"{col} -> {cardinality}")
            embeddings[col] = keras.layers.Embedding(
                input_dim=cardinality + 1, output_dim=10
            )(inputs[col])

        embeddings_concatenated = tf.keras.layers.Concatenate()(list(embeddings.values()))
        embeddings_concatenated = tf.keras.layers.Flatten()(embeddings_concatenated)

        # continuous inputs concatenation
        continuous_concatenated = tf.keras.layers.Concatenate()(
            [inputs[col] for col in self.continuous_cols]
        )

        all_concatenated = tf.keras.layers.Concatenate()(
            [embeddings_concatenated, continuous_concatenated]
        )

        # dense layers
        bottleneck = tf.keras.layers.Dense(units=16, activation="linear")(
            all_concatenated
        )

        output_continuous = tf.keras.layers.Dense(
            units=len(continuous_cols), name="output_cont"
        )(bottleneck)

        output_categorical = {
            col: tf.keras.layers.Dense(
                units=df[col].nunique(), activation="softmax", name=f"output_{col}"
            )(bottleneck)
            for col in embedding_cols
        }

        outputs = {k: v for k, v in output_categorical.items()}
        outputs["output_cont"] = output_continuous
        print(outputs)
        model = tf.keras.models.Model(
            inputs=list(inputs.values()),
            outputs=outputs,
        )

        encoder = tf.keras.models.Model(inputs=list(inputs.values()), outputs=bottleneck)
        decoder = tf.keras.models.Model(inputs=bottleneck, outputs=outputs)

        return model, encoder, decoder

    def get_compiled_models(self, optim="adam", loss="mse"):
        model, encoder, decoder = self._build_models()

        losses = {
            f"output_{col}": tf.keras.losses.CategoricalCrossentropy()
            for col in embedding_cols
        }
        losses["output_cont"] = "mse"
        print(losses)
        model.compile(
            optimizer=optim,
            loss=losses,
        )

        return model, encoder, decoder


#%%
model, encoder, decoder = NNModel(embedding_cols, continuous_cols).get_compiled_models(
    optim=tf.keras.optimizers.Adam(learning_rate=0.0001)
)
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)


#%%
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs/fit/", histogram_freq=1
)
model.fit(
    dataset.shuffle(256).batch(16).prefetch(5),
    epochs=100,
    callbacks=[tensorboard_callback],
)


#%%
original = list(dataset.batch(1).take(1))[0][0]
representation = encoder(original)
reconstructed = decoder(representation)
for t_o, t_r in zip(original.values(), reconstructed.values()):
    print(t_o, t_r)
    print("-" * 20)

print(representation)