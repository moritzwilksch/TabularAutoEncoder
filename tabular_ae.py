#%%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from tensorflow import keras

#%%
class DataLoader:
    def __init__(self, dataset_name: str, embedding_cols: list, continuous_cols: list):
        self.dataset = dataset_name
        self.embedding_cols = embedding_cols
        self.continuous_cols = continuous_cols

    def _load_raw(self):
        return sns.load_dataset(self.dataset)

    def _process(self, data: pd.DataFrame):

        # cat to integer
        self.oe = OrdinalEncoder(dtype=np.int8)
        data[self.embedding_cols] = self.oe.fit_transform(data[self.embedding_cols])

        # scale continuous
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        data[self.continuous_cols] = self.scaler.fit_transform(data[self.continuous_cols])

        return data

    def load(self):
        raw = self._load_raw()
        processed = self._process(raw)
        return processed


#%%
embedding_cols = ["sex", "smoker", "day", "time"]
continuous_cols = ["total_bill", "size", "tip"]

df = DataLoader(
    "tips", embedding_cols=embedding_cols, continuous_cols=continuous_cols
).load()

train, validation, _, _ = train_test_split(df, df, test_size=0.2, random_state=42)


#%%
def dataset_from_dataframe(data: pd.DataFrame) -> tf.data.Dataset:
    data = data.copy()
    target_data = {col: data[col] for col in embedding_cols}
    target_data["output_cont"] = data[continuous_cols]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {col: data[col] for col in embedding_cols + continuous_cols},
            {col: data[col] for col in embedding_cols + continuous_cols},
            # target_data,
        )
    )

    return dataset


cardinalities = df.nunique().to_dict()
dataset = dataset_from_dataframe(train)
validation_dataset = dataset_from_dataframe(validation)


class TabularAE(keras.Model):
    def __init__(self, embedding_cols: list, continuous_cols: list, cardinalities: dict):
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
            col: tf.keras.layers.Dense(units=1, activation="linear", name=f"output_{col}")
            for col in self.continuous_cols
        }

        # categorical
        self.output_categorical = {
            col: tf.keras.layers.Dense(
                units=self.cardinalities[col], activation="softmax", name=f"output_{col}"
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

        outputs = {colname: layer(bottleneck) for colname, layer in self.outputs.items()}

        return outputs


    def decode(self, bottleneck):
        outputs = {colname: layer(bottleneck) for colname, layer in self.outputs.items()}

        return outputs


model = TabularAE(embedding_cols, continuous_cols, cardinalities)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs/fit/", histogram_freq=1
)

losses = {
    col: tf.keras.losses.SparseCategoricalCrossentropy()
    if col in embedding_cols
    else "mse"
    for col in embedding_cols + continuous_cols
}

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=losses,
)

model.fit(
    dataset.shuffle(256).batch(32).prefetch(5),
    validation_data=validation_dataset.batch(512).prefetch(5),
    epochs=250,
    callbacks=[tensorboard_callback],
)


#%%
original = list(dataset.batch(1).take(1))[0][0]
print(original)

representation = model(original, return_bottleneck=True)
reconstructed = model.decode(representation)
for t_o, t_r in zip(original.values(), reconstructed.values()):
    print("-" * 20)
    print(t_o, t_r)
print("=" * 20)
print(f"REPRESENTATION: {representation}")
