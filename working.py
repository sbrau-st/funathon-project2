# %%
import mlflow
from dotenv import load_dotenv
import polars as pl
from sklearn.model_selection import train_test_split

load_dotenv(override=True)

df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/"
                     "diffusion/funathon/2026/project2/generation_None_temp08.parquet")
print(df.head())
print(f"Total rows: {len(df)}")

n_classes = df['code'].n_unique()
print(f"Number of unique NACE codes: {n_classes}")
# %%

# %%

train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
