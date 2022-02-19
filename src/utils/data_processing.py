import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


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
        data[self.continuous_cols] = self.scaler.fit_transform(
            data[self.continuous_cols]
        )

        return data

    def load(self):
        raw = self._load_raw()
        processed = self._process(raw)
        return processed


def dataset_from_dataframe(
    data: pd.DataFrame, embedding_cols: list, continuous_cols: list
) -> tf.data.Dataset:
    data = data.copy()
    target_data = {col: data[col] for col in embedding_cols}
    target_data["output_cont"] = data[continuous_cols]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {col: data[col] for col in embedding_cols + continuous_cols},
            {col: data[col] for col in embedding_cols + continuous_cols},
        )
    )

    return dataset
