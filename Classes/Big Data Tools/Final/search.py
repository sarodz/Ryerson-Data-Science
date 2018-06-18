import argparse
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import sum, column, desc 
from pyspark.sql.types import IntegerType

def main(Number, Query, File, write=False):
    print("\nSetting up the enviroment\n")
    conf = SparkConf().setAppName("Searcher") # configure Spark
    sc = SparkContext(conf = conf)    # start Spark Context with the specific configuration
    sql = SQLContext(sc) # start Spark SQL
    
    print("\nLoading the data\n")
    data = sql.read.load(File)
        
    print("\nQuerying the keywords in the database\n")
    totKeyword = len(Query)
    filtered = data.filter(column('word').isin([word.lower() for word in Query])) # Query the database based on request
    
    sumed = filtered.groupby(filtered.location).agg(sum('tfIdf').alias("tot"))  # Sum the TFIDF scores
    
    counted = filtered.groupby(filtered.location).count()
    counted = counted.select(counted.location.alias("loc"), (column("count")/totKeyword).alias("freq")) # determine the weight for each word
    
    result = sumed.join(counted, on = sumed.location==counted.loc, how = "inner") # join the tables
    result = result.select(result.location, (column("tot") * column("freq")).alias("score")).orderBy(desc("score")).limit(Number) # Calculate score and return top N values
    
    if write:
        print("\nWriting the data\n")
        result.write.format('com.databricks.spark.csv').save('query_'+''.join(Query), header='true')
    else:
        result.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="searcher", add_help=False)
    parser.add_argument("-N", default = 1, type = int, 
                        help= 'Picks the top N documents that match the query')
    parser.add_argument("-Q", type= str, required=True, nargs='*',
                        help= 'User provided query')
    parser.add_argument("-F", type= str, default='bbc.parquet',
                        help= 'Name of file. It is assumed that it is located in the same location as this file')
    input = parser.parse_args()
    main(input.N, input.Q, input.F, write=False)