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

from elephas.ml_model import ElephasEstimator
from elephas import optimizers as elephas_optimizers
from elephas.ml.adapter import from_data_frame, to_data_frame


#limit to CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from pyspark.context import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)



image_df = (sqlContext
             .read
             .parquet("data/idigbio-media-20171112T013207-mercury-images-32x32.parquet")
             .limit(100)
             .filter(col("imgnpa").isNotNull())
           )

print(image_df.count())

# Scaling goes here

renamed_df = image_df.withColumn("features", col("imgnpa")).withColumn("label", col("contaminated"))

# Split in to train/test
splits = renamed_df.randomSplit([0.8, 0.2], 314)
train_df = splits[0]
test_df = splits[1]

print(train_df.count())
print(test_df.count())


features_train, labels_train = from_data_frame(train_df, True, 2)
features_test, labels_test = from_data_frame(test_df, True, 2)

import numpy as np
from math import sqrt
# Change features to right shape
def raise_dim(l):
    x = np.array(l)
    square_dim = int(sqrt(len(x))) # gave up, hard coded
    r = x[0::3].reshape(-1, 32)
    g = x[1::3].reshape(-1, 32)
    b = x[2::3].reshape(-1, 32)
    return np.array([r, g, b])
    # np.expand_dims(, axis=0)

#def one_hot(l):
#    return 1 if l else 0

features_train = np.array([raise_dim(x) for x in features_train])
features_test = np.array([raise_dim(x) for x in features_test])

#from keras.utils import to_categorical
#labels_train = to_categorical(labels_train)
#labels_test = to_categorical(labels_test)

print(features_train[0].shape)

# Model
K.set_image_dim_ordering('th')

width, height = 32, 32
if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, width, height)))
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
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



# Define elephas optimizer (which tells the model how to aggregate updates on the Spark master)
adadelta = elephas_optimizers.Adadelta()

from elephas.utils.rdd_utils import to_labeled_point
from elephas.utils.rdd_utils import to_simple_rdd
lp_rdd = to_simple_rdd(sc, features_train, labels_train)

#print(lp_rdd.take(5))


from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers

adagrad = elephas_optimizers.Adagrad()
spark_model = SparkModel(sc,model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=8)
spark_model.train(lp_rdd, nb_epoch=20, batch_size=32, verbose=0, validation_split=0.1)

print(spark_model)



prediction = spark_model.predict_classes(features_test)
print(prediction)
truth = [l[1] for l in labels_test]

from sklearn.metrics import confusion_matrix
print(confusion_matrix(truth, prediction))

