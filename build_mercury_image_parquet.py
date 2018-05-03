
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
from imagehash import phash
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.linalg import Vectors

import numpy as np


from pyspark.ml.linalg import *
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.feature import StringIndexer, StandardScaler


from pyspark.context import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)



def image_nparray(v):
    try:
        resp = urlopen("http://api.idigbio.org/v2/media?size=webview&filereference=" + v)
        pilImage = Image.open(BytesIO(resp.read())).resize((256,256),Image.LANCZOS)
        array = np.asarray(pilImage).astype(float)
        array = array.ravel()
        return array.tolist()
    except:
        return np.nan


joined_df = sqlContext.read.parquet("/outputs/idigbio-media-20171112T013207-mercury-usecase.parquet")
imgnpa_udf = F.udf(image_nparray, T.ArrayType(T.FloatType()))
npdf = joined_df.select("accessuri","contaminated",imgnpa_udf("accessuri").alias("imgnpa"))


#to_vectors = F.udf(lambda x: DenseVector(x), VectorUDT())
#vector_df = npdf.withColumn("vector_images",to_vectors("image_nparray(accessuri)"))



#get_vector_length = F.udf(lambda vec : vec.size, T.IntegerType())
#vector_length_df = vector_df.withColumn("vector_length",get_vector_length("vector_images"))
##vector_length_df.describe().show()

#equal_df = vector_length_df.filter(F.col("vector_length")==196608)
#same_length_df = equal_df.select("vector_images","contaminated")

# Standardizes features by scaling to unit variance and/or removing the mean using column summary statistics on
# the samples in the training set
# Standardization can improve the convergence rate during the optimization process, and also prevents against
# features with very large variances exerting an overly large influence during model training.


#scaler = StandardScaler(inputCol="vector_images", outputCol="scaled_features", withStd=True, withMean=False)
##fitted_scaler = scaler.fit(same_length_df)
## scaled_df = fitted_scaler.transform(same_length_df)
#input_data = np.array(same_length_df.select('vector_images').collect())


(npdf
  .write
  .mode("overwrite")
  .parquet("/outputs/idigbio-media-20171112T013207-mercury-images.parquet")
)

