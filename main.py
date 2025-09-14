import pandas as pd

# Original laden
df = pd.read_csv("spam_big.csv", encoding="UTF-8")[["label", "text"]]
df.columns = ["label", "text"]

# 200-fach duplizieren (~1.1 Mio Zeilen)
df_big = pd.concat([df]*5, ignore_index=True)

# Optional: Reihenfolge mischen
df_big = df_big.sample(frac=1).reset_index(drop=True)

# Speichern
df_big.to_csv("spam_bigger.csv", index=False)
