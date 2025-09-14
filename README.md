# Naive Bayes for Big Data Processing

Dieses Repository enthält eine Fallstudie zur Anwendung von Naive Bayes im Bereich Big Data Processing. Ziel ist der Vergleich einer klassischen scikit-learn Implementierung (Single Node) mit einer verteilten Apache Spark MLlib Implementierung. Es wurde das SMS Spam Collection Dataset unter https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data verwendet. 

# Inhalte

- Datengrundlage: SMS Spam Collection Dataset (dupliziert für größere Datenmengen)
- Implementierung mit Python & scikit-learn
- Skalierungsexperimente mit Apache Spark MLlib
- Vergleich von Laufzeiten und Klassifikationsgenauigkeit
- Poster & Dokumentation der Ergebnisse

# Ergebnisse (Kurzfassung)
- scikit-learn: 98,9 % Accuracy, schnelle Trainingszeit, aber aufwändige Feature-Extraktion
- Spark MLlib: Laufzeitvorteile durch Parallelisierung, geringere Accuracy

# Ausführung
1. Nach Installation aller Paket erst die main.py zum duplizieren der Daten
2. Dann single_node.py für Einzelausführung auf dem Gerät mit scikit-learn
3. Anschließend cluster_comparision.py für Auswertung von Spark MLlib als Single-Node und Mulit-Node
