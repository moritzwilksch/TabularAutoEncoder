import time

import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

from src.models.tabular_ae import TabularAE
from src.utils.data_processing import DataLoader, dataset_from_dataframe

# -------------------------- Load config --------------------------
with open("src/models/config.yaml") as f:
    config = yaml.safe_load(f)

EMBEDDING_COLS = config.get("data_config").get("embedding_cols")
CONTINUOUS_COLS = config.get("data_config").get("continuous_cols")
BOTTLENECK_DIM = config.get("model_config").get("architecture").get("bottleneck_dim")
EMBEDDING_DIM = config.get("model_config").get("architecture").get("embedding_dim")

df = DataLoader(
    "tips", embedding_cols=EMBEDDING_COLS, continuous_cols=CONTINUOUS_COLS
).load()

train, validation, _, _ = train_test_split(df, df, test_size=0.2, random_state=42)


cardinalities = df.nunique().to_dict()
dataset = dataset_from_dataframe(
    train, embedding_cols=EMBEDDING_COLS, continuous_cols=CONTINUOUS_COLS
)
validation_dataset = dataset_from_dataframe(
    validation, embedding_cols=EMBEDDING_COLS, continuous_cols=CONTINUOUS_COLS
)

model = TabularAE(
    EMBEDDING_COLS,
    CONTINUOUS_COLS,
    cardinalities,
    embedding_dim=EMBEDDING_DIM,
    bottleneck_dim=BOTTLENECK_DIM,
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs/fit/", histogram_freq=1
)

losses = {
    col: tf.keras.losses.SparseCategoricalCrossentropy()
    if col in EMBEDDING_COLS
    else "mse"
    for col in EMBEDDING_COLS + CONTINUOUS_COLS
}

loss_weights = {
    col: 1 if col in EMBEDDING_COLS else 2 for col in EMBEDDING_COLS + CONTINUOUS_COLS
}

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=losses,
    loss_weights=loss_weights,
)

tic = time.perf_counter()
model.fit(
    dataset.shuffle(256).batch(32).prefetch(5),
    validation_data=validation_dataset.batch(512).prefetch(5),
    epochs=250,
    callbacks=[tensorboard_callback],
)
tac = time.perf_counter()
print(f"Fitting took {tac-tic:.1f} seconds.")
model.save_weights("src/models/saved/tabae.h5")


original = list(dataset.batch(1).take(1))[0][0]
print(original)

representation = model(original, return_bottleneck=True)
reconstructed = model.decode(representation)
for t_o, t_r in zip(original.values(), reconstructed.values()):
    print("-" * 20)
    print(t_o, t_r)
print("=" * 20)
print(f"REPRESENTATION: {representation}")
