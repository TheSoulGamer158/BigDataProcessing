# Naive Bayes for Big Data Processing

Dieses Repository enthält eine Fallstudie zur Anwendung von Naive Bayes im Bereich Big Data Processing. Ziel ist der Vergleich einer klassischen scikit-learn Implementierung (Single Node) mit einer verteilten Apache Spark MLlib Implementierung.

Inhalte

Datengrundlage: SMS Spam Collection Dataset (dupliziert für größere Datenmengen)

Implementierung mit Python & scikit-learn

Skalierungsexperimente mit Apache Spark MLlib

Vergleich von Laufzeiten und Klassifikationsgenauigkeit

Poster & Dokumentation der Ergebnisse

Ergebnisse (Kurzfassung)

scikit-learn: 98,9 % Accuracy, schnelle Trainingszeit, aber aufwändige Feature-Extraktion

Spark MLlib: Laufzeitvorteile durch Parallelisierung, geringere Accuracy
