import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.functions import when
import time
from pyspark.sql.functions import col


# 1. JAVA_HOME korrekt setzen
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.16.8-hotspot'
os.environ['SPARK_LOCAL_IP'] = 'localhost'

# 2. Spark Session starten
spark = SparkSession.builder \
    .appName("NaiveBayesSpam") \
    .master("local[4]") \
    .config("spark.driver.host", "localhost") \
    .config("spark.driver.bindAddress", "localhost") \
    .getOrCreate()

print("Spark Version:", spark.version)



# 2. CSV in Spark laden
df = spark.read.csv("spam_bigger.csv", header=True, inferSchema=True)

# 2.1 Null Texte ausfiltern
df = df.filter(col("text").isNotNull())

# 3. Labels in Double konvertieren (Spark NB braucht Double)
df = df.withColumn(
    "label",
    when(col("label") == "spam", 1.0).when(col("label") == "ham", 0.0)
)

# 4. Train-Test Split (Spark)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 5. Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
train_df = tokenizer.transform(train_df)
test_df = tokenizer.transform(test_df)

# 6. Stopwords entfernen
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
train_df = remover.transform(train_df)
test_df = remover.transform(test_df)

# 7. CountVectorizer + IDF (TF-IDF)
vectorizer = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=2500)
vectorizer_model = vectorizer.fit(train_df)
train_df = vectorizer_model.transform(train_df)
test_df = vectorizer_model.transform(test_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(train_df)
train_df = idf_model.transform(train_df)
test_df = idf_model.transform(test_df)

# 8. Naive Bayes Training
nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")

t0 = time.time()
model = nb.fit(train_df)
train_time = time.time() - t0

# 9. Evaluation
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Naive Bayes Training & Transformation: {train_time:.2f} Sekunden")

# 10. Spark Session stoppen
spark.stop()
