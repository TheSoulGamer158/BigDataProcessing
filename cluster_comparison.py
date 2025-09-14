import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------------------
# 1. Java & Spark Session
# ---------------------------
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.16.8-hotspot'
os.environ['SPARK_LOCAL_IP'] = 'localhost'

spark = SparkSession.builder \
    .appName("NaiveBayesSpam") \
    .master("local[32]") \
    .config("spark.driver.host", "localhost") \
    .config("spark.driver.bindAddress", "localhost") \
    .getOrCreate()

print("Spark Version:", spark.version)

# ---------------------------
# 2. CSV laden und vorbereiten
# ---------------------------
df = spark.read.csv("spam_bigger.csv", header=True, inferSchema=True)
df = df.filter(col("text").isNotNull())
indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
df = indexer.fit(df).transform(df)

# ---------------------------
# 3. Single Node vs Multi-Partition
# ---------------------------
single_node_df = df.repartition(1)
multi_partition_df = df.repartition(32)

# ---------------------------
# 4. Funktionen für Feature-Aufbereitung
# ---------------------------
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

def extract_features(train_df, test_df):
    # Tokenize & StopWords entfernen
    train_token = remover.transform(tokenizer.transform(train_df))
    test_token = remover.transform(tokenizer.transform(test_df))

    # CountVectorizer auf Train fitten, Test transformieren
    vectorizer = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=2500)
    cv_model = vectorizer.fit(train_token)
    train_vector = cv_model.transform(train_token)
    test_vector = cv_model.transform(test_token)

    # IDF auf Train fitten, Test transformieren
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(train_vector)
    train_features = idf_model.transform(train_vector)
    test_features = idf_model.transform(test_vector)

    return train_features, test_features

# ---------------------------
# 5. Funktion für Naive Bayes Training & Evaluation
# ---------------------------
def train_nb(df_input):
    train_df, test_df = df_input.randomSplit([0.8, 0.2], seed=42)

    # Feature-Extraktion messen
    t0_feat = time.time()
    train_feat, test_feat = extract_features(train_df, test_df)
    feat_time = time.time() - t0_feat

    # Naive Bayes Training messen
    nb = NaiveBayes(featuresCol="features", labelCol="labelIndex", modelType="multinomial")
    t0_train = time.time()
    model = nb.fit(train_feat)
    train_time = time.time() - t0_train

    predictions = model.transform(test_feat)
    evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    return accuracy, feat_time, train_time

# ---------------------------
# 6. Single Node Training
# ---------------------------
acc_single, feat_single, train_single = train_nb(single_node_df)
print(f"[Single Node] Accuracy: {acc_single:.4f}, Feature-Extraktion: {feat_single:.2f}s, Training: {train_single:.2f}s")

# ---------------------------
# 7. Multi-Partition Training
# ---------------------------
acc_multi, feat_multi, train_multi = train_nb(multi_partition_df)
print(f"[Multi-Partition] Accuracy: {acc_multi:.4f}, Feature-Extraktion: {feat_multi:.2f}s, Training: {train_multi:.2f}s")

# ---------------------------
# 8. Spark Session stoppen
# ---------------------------
spark.stop()
