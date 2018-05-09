activate_this_file = "/home/mcollins/spark_keras2/bin/activate_this.py"
with open(activate_this_file) as f:
    exec(f.read(), {'__file__': activate_this_file})


import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql import types as T
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.feature import StringIndexer, StandardScaler

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# Ended up with multiple elephas libraries:
#import importlib.util
#spec = importlib.util.spec_from_file_location("elephas", "/home/mcollins/spark_keras2/lib/python3.6/site-packages/elephas-0.3-py3.6.egg/elephas/__init__.py")
#e = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(e)

# Fixed this by exporting pyspark pthon version to be the one inside the virtualenv
#print(sys.path)
#orig_path = sys.path
#sys.path = ['/home/mcollins/spark_keras2/lib/python3.6/site-packages', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/elephas-0.3-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/Flask-1.0.2-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/itsdangerous-0.24-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/click-6.7-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/Werkzeug-0.14.1-py3.6.egg', '/home/mcollins/pragma-cloud-spark', '/opt/spark/2.3.0/python/lib/pyspark.zip', '/opt/spark/2.3.0/python/lib/py4j-0.10.6-src.zip', '/opt/spark/2.3.0/jars/spark-core_2.11-2.3.0.jar', '/opt/python/lib/python36.zip', '/opt/python/lib/python3.6', '/opt/python/lib/python3.6/lib-dynload']
#sys.path = ['/home/mcollins/spark_keras2/lib/python3.6/site-packages', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/elephas-0.3-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/Flask-1.0.2-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/itsdangerous-0.24-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/click-6.7-py3.6.egg', '/home/mcollins/spark_keras2/lib/python3.6/site-packages/Werkzeug-0.14.1-py3.6.egg', '/home/mcollins/pragma-cloud-spark']
from elephas.ml_model import ElephasEstimator
from elephas import optimizers as elephas_optimizers
from elephas.ml.adapter import from_data_frame, to_data_frame


#sys.path = orig_path

#limit to CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from pyspark.context import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)



image_df = sqlContext.read.parquet("data/idigbio-media-20171112T013207-mercury-images-100.parquet")

print(image_df.count())


# Convert nparrays to vectors for models
#to_vectors = F.udf(lambda x: DenseVector(x), VectorUDT())
#vector_df = image_df.withColumn("vector_images", to_vectors(col("imgnpa")))

# Let's try just numpy arrays!
#vector_df = image_df.select(col("contaminated"), col("imgnpa").alias("vector_images"))


#vector_df = to_data_frame(sc, image_df["


# And verify they're all the same size
#get_vector_length = F.udf(lambda vec : len(vec), T.IntegerType()) # vec.size for DenseVector
#vector_length_df = vector_df.withColumn("vector_length", get_vector_length("vector_images"))
#
#(vector_length_df
#  .select(F.max(col("vector_length")), F.min(col("vector_length")))
#  .show()
#)


# Scale features

#scaler = StandardScaler(inputCol="vector_images", outputCol="scaled_features", withStd=True, withMean=False)
#fitted_scaler = scaler.fit(vector_length_df)
#scaled_df = fitted_scaler.transform(vector_length_df)

#import numpy as np
#from sklearn import preprocessing
#scale = F.udf(lambda x: x / 255.0, T.Vec)
#scaled_df = vector_length_df.withColumn("scaled_features", scale(col("vector_images")))

#features is a magic name in the prediction code
#scaled_df = vector_length_df.withColumn("features", col("vector_images"))
scaled_df = image_df.withColumn("features", col("imgnpa")).withColumn("label", col("contaminated"))

features, labels = from_data_frame(scaled_df, True, 2)

import numpy as np
# Change features to right shape
def raise_dim(l):
    x = np.array(l)
    r = x[0::3].reshape(-1, 2)
    g = x[1::3].reshape(-1, 2)
    b = x[2::3].reshape(-1, 2)
    return np.expand_dims(np.array([r, g, b]), axis=0)

f2 = np.array([raise_dim(x) for x in features])

lb_df = to_data_frame(sc, f2, labels, True)
lb_df.printSchema()
lb_df.show()

## Turn back mllib vectors
#from pyspark.mllib import 
#to_mllib_vectors = F.udf(lambda x: 


#scaled_df.show()


# Split in to train/test
splits = lb_df.randomSplit([0.8, 0.2], 314)
train_df = splits[0]
test_df = splits[1]

print(train_df.count())
print(test_df.count())

train_df.show()

# Model

width, height = 32, 32
if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



# Define elephas optimizer (which tells the model how to aggregate updates on the Spark master)
adadelta = elephas_optimizers.Adadelta()

# Initialize SparkML Estimator and set all relevant properties
estimator = ElephasEstimator()
estimator.setFeaturesCol("features")             # These two come directly from pyspark,
estimator.setLabelCol("label")                 # hence the camel case. Sorry :)
estimator.set_keras_model_config(model.to_yaml())       # Provide serialized Keras model
estimator.set_optimizer_config(adadelta.get_config())   # Provide serialized Elephas optimizer
estimator.set_categorical_labels(True)
estimator.set_nb_classes(2)
estimator.set_num_workers(1)  # We just use one worker here. Feel free to adapt it.
estimator.set_nb_epoch(20) 
estimator.set_batch_size(128)
estimator.set_verbosity(1)
estimator.set_validation_split(0.15)


fitted_model = estimator.fit(train_df)
prediction = fitted_model.transform(test_df)
pnl = prediction.select("label", "prediction")
pnl.show(100)


#from pyspark.ml import Pipeline

#pipeline = Pipeline(stages=[estimator])



#fitted_pipeline = pipeline.fit(train_df) # Fit model to data

#prediction = fitted_pipeline.transform(test_df)

#pnl = prediction.select("index_category", "prediction")
#pnl.show(100)



