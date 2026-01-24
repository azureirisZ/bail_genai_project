import pandas as pd

df = pd.read_parquet(
    r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_case_factor_matrix.parquet"
)

print(df.shape)
print(df.head())
