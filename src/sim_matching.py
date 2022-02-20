#%%
import numpy as np
from sklearn import neighbors
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

cardinalities = df.nunique().to_dict()

#%%
model = TabularAE(EMBEDDING_COLS, CONTINUOUS_COLS, cardinalities, embedding_dim=EMBEDDING_DIM, bottleneck_dim=BOTTLENECK_DIM)
model.build({col: (1, ) for col in df.columns})
model.load_weights("src/models/saved/tabae.h5")

#%%
records = df.to_dict(orient="records")

vectors = []
for rec in records:
    rec = {col: tf.reshape(tf.convert_to_tensor(val), (-1, 1)) for col, val in rec.items()}
    pred = model.call(rec, return_bottleneck=True).numpy()
    vectors.append(pred)

#%%
mtx = np.vstack(vectors)
#%%
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors()
nn.fit(mtx)

dists, neighbors = nn.kneighbors(mtx[1, :].reshape(1, -1))

df.iloc[neighbors[0], :]
