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
