#https://github.com/maxpumperla/elephas/blob/master/examples/Spark_ML_Pipeline.ipynb

import os
import sys

activate_this_file = "/home/mcollins/spark_keras2/bin/activate_this.py"
with open(activate_this_file) as f:
    exec(f.read(), {'__file__': activate_this_file})

from pyspark.context import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)
print(sqlContext)


from pyspark.ml.linalg import Vectors
import numpy as np
import random

sql_context = SQLContext(sc)

data_path = "data"


# This happens locally
def shuffle_csv(csv_file):
    lines = open(csv_file).readlines()
    random.shuffle(lines)
    open(csv_file, 'w').writelines(lines)

def load_data_frame(csv_file, shuffle=True, train=True):
    fn = os.path.join(data_path, csv_file)
    if shuffle:
        shuffle_csv(fn)
    data = sc.textFile(fn) # This is an RDD, which will later be transformed to a data frame
    data = data.filter(lambda x:x.split(',')[0] != 'id').map(lambda line: line.split(','))
    if train:
        data = data.map(
            lambda line: (Vectors.dense(np.asarray(line[1:-1]).astype(np.float32)),
                          str(line[-1])) )
    else:
        # Test data gets dummy labels. We need the same structure as in Train data
        data = data.map( lambda line: (Vectors.dense(np.asarray(line[1:]).astype(np.float32)),"Class_1") ) 
    return sqlContext.createDataFrame(data, ['features', 'category'])

train_df = load_data_frame("train.csv")
test_df = load_data_frame("test.csv", shuffle=False, train=False) # No need to shuffle test data

print("Train data frame:")
train_df.show(10)

print("Test data frame (note the dummy category):")
test_df.show(10)


from pyspark.ml.feature import StringIndexer

string_indexer = StringIndexer(inputCol="category", outputCol="index_category")
fitted_indexer = string_indexer.fit(train_df)
indexed_df = fitted_indexer.transform(train_df)

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
fitted_scaler = scaler.fit(indexed_df)
scaled_df = fitted_scaler.transform(indexed_df)

print("The result of indexing and scaling. Each transformation adds new columns to the data frame:")
scaled_df.show(10)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, generic_utils

nb_classes = train_df.select("category").distinct().count()
input_dim = len(train_df.select("features").first()[0])

model = Sequential()
model.add(Dense(512, input_shape=(input_dim,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')




from elephas.ml_model import ElephasEstimator
from elephas import optimizers as elephas_optimizers

# Define elephas optimizer (which tells the model how to aggregate updates on the Spark master)
adadelta = elephas_optimizers.Adadelta()

# Initialize SparkML Estimator and set all relevant properties
estimator = ElephasEstimator()
estimator.setFeaturesCol("scaled_features")             # These two come directly from pyspark,
estimator.setLabelCol("index_category")                 # hence the camel case. Sorry :)
estimator.set_keras_model_config(model.to_yaml())       # Provide serialized Keras model
estimator.set_optimizer_config(adadelta.get_config())   # Provide serialized Elephas optimizer
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)
estimator.set_num_workers(1)  # We just use one worker here. Feel free to adapt it.
estimator.set_nb_epoch(20) 
estimator.set_batch_size(128)
estimator.set_verbosity(1)
estimator.set_validation_split(0.15)




from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[string_indexer, scaler, estimator])


fitted_pipeline = pipeline.fit(train_df) # Fit model to data

prediction = fitted_pipeline.transform(train_df) # Evaluate on train data.
# prediction = fitted_pipeline.transform(test_df) # <-- The same code evaluates test data.
pnl = prediction.select("index_category", "prediction")
pnl.show(100)

prediction_and_label = pnl.map(lambda row: (row.index_category, row.prediction))

prediction_and_label.head(10)





#from pyspark.mllib.evaluation import MulticlassMetrics
#metrics = MulticlassMetrics(prediction_and_label)
#print(metrics.precision())

