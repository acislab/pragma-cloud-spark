import os
import sys
from pyspark.context import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)
print(sqlContext)


from pyspark.mllib.linalg import Vectors
import numpy as np
import random

sql_context = SQLContext(sc)

data_path = "data"

df = sc.textFile(os.path.join(data_path, "test.csv"))

print(df.count())

sys.exit(0)

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
