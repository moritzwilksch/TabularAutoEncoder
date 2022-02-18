#%%
import numpy as np
import pandas as pd
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
dataset = tf.data.Dataset.from_tensor_slices(
    ({col: df[col] for col in embedding_cols + continuous_cols}, df["tip"])
)


#%%
class NNModel:
    def __init__(self, embedding_cols: list, continuous_cols: list):
        self.embedding_cols = embedding_cols
        self.continuous_cols = continuous_cols

    def _build_model(self):
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

        embeddings_concatenated = tf.keras.layers.Concatenate()(
            list(embeddings.values())
        )
        embeddings_concatenated = tf.keras.layers.Flatten()(embeddings_concatenated)

        # continuous inputs concatenation
        continuous_concatenated = tf.keras.layers.Concatenate()(
            [inputs[col] for col in self.continuous_cols]
        )

        all_concatenated = tf.keras.layers.Concatenate()(
            [embeddings_concatenated, continuous_concatenated]
        )

        # dense layers
        output_layer = tf.keras.layers.Dense(units=64, activation="linear")(
            all_concatenated
        )

        model = tf.keras.models.Model(
            inputs=list(inputs.values()), outputs=output_layer
        )

        return model

    def get_compiled_model(self, optim="adam", loss="mse"):
        model = self._build_model()
        model.compile(optim, loss)
        return model


#%%
model = NNModel(embedding_cols, continuous_cols).get_compiled_model(
    optim=tf.keras.optimizers.Adam(lr=10e-3)
)
print(model.summary())
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
#%%
model.fit(dataset.shuffle(256).batch(16).prefetch(5), epochs=250)
