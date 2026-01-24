import pandas as pd

df = pd.read_parquet(
    r"C:\Users\Sastra\OneDrive\Desktop\bail_genai_project\data\final\sc_bail_reason_sentences_clean.parquet"
)

print(df.head(10))
