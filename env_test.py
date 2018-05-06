import sys
import os

activate_this_file = "/home/mcollins/spark_keras2/bin/activate_this.py"
with open(activate_this_file) as f:
    exec(f.read(), {'__file__': activate_this_file})


#os.environ["PYSPARK_PYTHON"] = "spark_keras2/bin/python"
print(sys.version)


from pyspark.context import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)
print(sqlContext)


#sc.addPyFile("spark_keras2.zip")

#sys.path.insert(0, "deps.zip")

import h5py

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
model = Sequential()


# Spark work test

import random
NUM_SAMPLES = 1000
def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1

count = sc.parallelize(range(0, NUM_SAMPLES)) \
             .filter(inside).count()

print("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))
