import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.models.tabular_ae import TabularAE
from src.utils.data_processing import DataLoader, dataset_from_dataframe

embedding_cols = ["sex", "smoker", "day", "time"]
continuous_cols = ["total_bill", "size", "tip"]

df = DataLoader(
    "tips", embedding_cols=embedding_cols, continuous_cols=continuous_cols
).load()

train, validation, _, _ = train_test_split(df, df, test_size=0.2, random_state=42)


cardinalities = df.nunique().to_dict()
dataset = dataset_from_dataframe(
    train, embedding_cols=embedding_cols, continuous_cols=continuous_cols
)
validation_dataset = dataset_from_dataframe(
    validation, embedding_cols=embedding_cols, continuous_cols=continuous_cols
)

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


original = list(dataset.batch(1).take(1))[0][0]
print(original)

representation = model(original, return_bottleneck=True)
reconstructed = model.decode(representation)
for t_o, t_r in zip(original.values(), reconstructed.values()):
    print("-" * 20)
    print(t_o, t_r)
print("=" * 20)
print(f"REPRESENTATION: {representation}")