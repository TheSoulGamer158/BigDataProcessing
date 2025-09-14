import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur Datei anpassen
file_path = "spam.csv"

# Annahme: Datei ist CSV mit Spalten "label" und "text"
df = pd.read_csv(file_path, encoding="ANSI")

# Label-Verteilung zählen
label_counts = df['label'].value_counts()
total = label_counts.sum()

# Labels vorbereiten mit Anzahl und Prozent
labels = [f"{label} ({count:,} | {count/total:.1%})"
          for label, count in label_counts.items()]

# Plot erstellen
plt.figure(figsize=(6,6))
plt.pie(
    label_counts,
    labels=labels,
    startangle=90,
    colors=['#4caf50', '#f44336'],
    wedgeprops={'edgecolor': 'white'}
)

plt.title("Verteilung der Labels im SMS Spam Dataset")
plt.tight_layout()

# Plot speichern
plt.savefig("label_distribution_pie_counts_org.png", dpi=300)

plt.show()
