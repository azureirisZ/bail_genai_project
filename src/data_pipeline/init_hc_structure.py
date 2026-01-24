import os

hcs = ["kerala", "madras", "bombay", "allahabad", "rajasthan", "delhi"]

for hc in hcs:
    os.makedirs(f"src/data_pipeline/hc/{hc}", exist_ok=True)

os.makedirs("data/raw", exist_ok=True)

print("✅ HC folder structure created")
