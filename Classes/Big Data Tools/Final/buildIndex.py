# spark-submit --master yarn --executor-memory 512m --num-executors 3 --executor-cores 1 --driver-memory 512m buildIndex.py

import re
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import explode, sum, column, log
from pyspark.sql.types import IntegerType

def main():
    conf = SparkConf().setAppName("Index Builder") # configure Spark
    sc = SparkContext(conf = conf)   # start Spark Context with the specific configuration  
    sql = SQLContext(sc) # start Spark SQL

    text = sc.wholeTextFiles("/user/root/bbcsport/*") # fuzy read: Reads all files under bbcsport
    fileCount = text.count()
	# reformat data to make it cleaner and break text into words
    cleaned = text.map(lambda file: ("%s/%s" % (file[0].split("/")[len(file[0].split("/"))-2],\
                                        file[0].split("/")[len(file[0].split("/"))-1]), file[1].lower().split()))\
                  .map(lambda file: (file[0], [re.sub(r'[^\w\s]', '', word) for word in file[1]]))
    # regex cleaning from: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

    cleanedDF = cleaned.toDF(["location", "words"]) # create dataframe
    cleanedDF = cleanedDF.select(cleanedDF.location, explode(cleanedDF.words).alias("word")) # Flatten the list of words

    tfMap = cleanedDF.groupby(cleanedDF.location, cleanedDF.word).count() #Count occurences of a word in a document
    tfReduce = tfMap.groupby(tfMap.location, tfMap.word).agg(sum("count").alias("tf")) # Calculate TF

    idfMap = cleanedDF.distinct().groupby(cleanedDF.word).count() # count whether a word occured in a document
    idfReduce = idfMap.select(idfMap.word, log(fileCount/(column("count"))).alias("idf")) # Calculate IDF

    joinTfIdf = tfReduce.join(idfReduce, tfReduce.word == idfReduce.word, "inner") # Join TF & IDF tables   
    tfIdf = joinTfIdf.select(column("location"), tfReduce["word"], (column("tf") * column("idf")).alias("tfIdf")) # Calc. TFIDF
    
    tfIdf.write.parquet('bbc.parquet') # write file in an efficient file format

if __name__ == "__main__":
    main()