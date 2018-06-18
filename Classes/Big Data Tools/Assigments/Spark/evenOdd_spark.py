<!DOCTYPE html>
<html>
<head>
</head>
<body style="font-family: verdana, sans-serif;font-size: 12px;">
<p><strong>1. Streaming</strong></p>
<p>Execution info is on the slide</p>
<p><strong>2. Logistic Regression</strong></p>
<p><strong>Note: Fire up "pyspark" shell to execute commands in section 2-4</strong></p>
<p>&nbsp; &nbsp; &nbsp;&nbsp;<b>from </b>pyspark.mllib.classification <b>import </b>LogisticRegressionWithSGD, LogisticRegressionModel</p>
<p><b>from </b>pyspark.mllib.regression <b>import </b>LabeledPoint</p>
<p>data = [<br />&nbsp;&nbsp;&nbsp;&nbsp; LabeledPoint(0.0, [0.0, 1.0]),<br />&nbsp;&nbsp;&nbsp;&nbsp; LabeledPoint(1.0, [1.0, 0.0])</p>
<p>]</p>
<p>lrm = LogisticRegressionWithSGD.train(sc.parallelize(data), iterations=10)</p>
<p>lrm.predict([1.0, 0.0])</p>
<p>#Output: 1.0</p>
<p>lrm.predict([0.0, 1.0])</p>
<p>#Output: 0.0</p>
<p><i># </i><i>Save and load </i><i>model</i></p>
<p>lrm.save(sc, <b>"</b><b>lrsgd"</b>)</p>
<p>sameModel = LogisticRegressionModel.load(sc, <b>"</b><b>lrsgd"</b>)</p>
<p>sameModel.predict([1.0, 0.0])</p>
<p>sameModel.predict([0.0, 1.0])</p>
<p><br /><strong>3. KMeans</strong></p>
<p><b>from </b>pyspark.mllib.clustering <b>import </b>KMeans, KMeansModel</p>
<p>data = [[1.0,1.0],[1.0,0.8],[-1.0,1.0],[-1.0,-1.0]]</p>
<p>parsedData=sc.parallelize(data)</p>
<p>kmeansModel = KMeans.train(parsedData, 2, maxIterations=10, runs=10, initializationMode="random")</p>
<p>kmeansModel.predict([1.0, 1.0])</p>
<p>kmeansModel.predict([1.0, -2.0])</p>
<p>#the two points should be in different clusters</p>
<p># Save and load model</p>
<p>kmeansModel.save(sc, "KMeansModel")</p>
<p>model = KMeansModel.load(sc, "KMeansModel")</p>
<p>model.predict([1.0, 1.0])</p>
<p>model.predict([1.0, -2.0])</p>
<p><br /><strong>4. ALS</strong></p>
<p><b>from </b>pyspark.mllib.recommendation <b>import </b>ALS, MatrixFactorizationModel, Rating</p>
<p>r1 = (1, 1, 1.0)</p>
<p>r2 = (1, 2, 2.0)</p>
<p>r3 = (2, 1, 2.0)</p>
<p>ratings = sc.parallelize([r1, r2, r3])</p>
<p>model = ALS.train(ratings, 10, 10)</p>
<p># Evaluate the model on training data</p>
<p>testdata = sc.parallelize([(2,2),(1,1)])</p>
<p>predictions = model.predictAll(testdata)</p>
<p>predictions.collect()</p>
<p>#Save and load the model</p>
<p>model.save(sc, "myCollaborativeFilter")</p>
<p>sameModel = MatrixFactorizationModel.load(sc, "myCollaborativeFilter")</p>
<p>#Try using the loaded model</p>
<p>sameModel.predictAll(testdata).collect()</p>
<p></p>
<p><strong>5. PageRank</strong></p>
<p><strong></strong></p>
<p><strong>Using Python - GraphFrames API</strong></p>
<p>Update python to python 2.7 (Information on that is provided)</p>
<p>http://askubuntu.com/questions/101591/how-do-i-install-python-2-7-2-on-ubuntu</p>
<p>http://stackoverflow.com/questions/23302270/how-do-i-run-graphx-with-python-pyspark</p>
<p>Fire up pyspark:&nbsp;<code data-lang="bash">pyspark --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11</code></p>
<pre class="lang-py prettyprint prettyprinted"><code><span class="pln">from graphframes import *<br /><br />#Create a vertices dataframe<br /><br />v </span><span class="pun">=</span><span class="pln"> sqlContext</span><span class="pun">.</span><span class="pln">createDataFrame</span><span class="pun">([</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"a"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"Alice"</span><span class="pun">,</span><span class="pln"> </span><span class="lit">34</span><span class="pun">),</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"b"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"Bob"</span><span class="pun">,</span><span class="pln"> </span><span class="lit">36</span><span class="pun">),</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"c"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"Charlie"</span><span class="pun">,</span><span class="pln"> </span><span class="lit">30</span><span class="pun">),</span><span class="pln">
</span><span class="pun">],</span><span class="pln"> </span><span class="pun">[</span><span class="str">"id"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"name"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"age"</span><span class="pun">]) </span><span class="pln">

</span><span class="com"># Create an Edge DataFrame with "src" and "dst" columns</span><span class="pln">
e </span><span class="pun">=</span><span class="pln"> sqlContext</span><span class="pun">.</span><span class="pln">createDataFrame</span><span class="pun">([</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"a"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"b"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"friend"</span><span class="pun">),</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"b"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"c"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"follow"</span><span class="pun">),</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"c"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"b"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"follow"</span><span class="pun">),</span><span class="pln">
</span><span class="pun">],</span><span class="pln"> </span><span class="pun">[</span><span class="str">"src"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"dst"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"relationship"</span><span class="pun">])</span><span class="pln">
</span><span class="com"># Create a GraphFrame</span><span class="pln">
g </span><span class="pun">=</span><span class="pln"> </span><span class="typ">GraphFrame</span><span class="pun">(</span><span class="pln">v</span><span class="pun">,</span><span class="pln"> e</span><span class="pun">)<br />results = g.pageRank(resetProbability=0.15, tol=0.01)<br /></span></code>results.vertices().show()<br /><br /><br />Other Graph Algorithms:<br />1. <a href="http://go.databricks.com/hubfs/notebooks/3-GraphFrames-User-Guide-python.html">http://go.databricks.com/hubfs/notebooks/3-GraphFrames-User-Guide-python.html<br /></a>Instead of "display()" use "show()". Example: instead of display(result.vertices) use result.vertices.show()<br /><br />2. http://graphframes.github.io/api/python/graphframes.html?highlight=pagerank#graphframes.GraphFrame.pageRank<br />-----------------------</pre>
<p><strong>Using Scala - GraphX API. Python version is not available yet</strong></p>
<p><strong>Note: Fire up scala console using command "spark-shell" to execute section</strong></p>
<p><em>Copy the followers.txt and users.txt into hdfs</em></p>
<p>import org.apache.spark.graphx._</p>
<p>val graph = GraphLoader.edgeListFile(sc, "/user/root/followers.txt")</p>
<p>// Run PageRank</p>
<p>val ranks = graph.pageRank(0.0001).vertices</p>
<p>// Join the ranks with the usernames</p>
<p>val users = sc.textFile("/user/root/users.txt").map { line =&gt;</p>
<p>val fields = line.split(",")</p>
<p>(fields(0).toLong, fields(1))</p>
<p>}</p>
<p>val ranksByUsername = users.join(ranks).map {</p>
<p>case (id, (username, rank)) =&gt; (username, rank)</p>
<p>}</p>
<p>// Print the result</p>
<p>println(ranksByUsername.collect().mkString("\n"))</p>
<p></p>
<p><strong>6. DataFrames - SQLContext</strong></p>
<p><a href="http://hortonworks.com/apache/spark/#section_8">http://hortonworks.com/apache/spark/#section_8</a></p>
<p><a href="https://spark.apache.org/docs/1.6.0/sql-programming-guide.html#sql">https://spark.apache.org/docs/1.6.0/sql-programming-guide.html#sql</a></p>
<p>Copy files kv1.txt, people.txt, and people.json to hdfs location "/user/root"</p>
<p></p>
<p><strong>Convert RDDS into DataFrames</strong></p>
<pre><code class="language-python" data-lang="python"><span class="kn">from</span> <span class="nn">pyspark.sql</span> <span class="kn">import</span> <span class="n">SQLContext</span><span class="p">,</span> <span class="n">Row</span>
<span class="n">sqlContext</span> <span class="o">=</span> <span class="n">SQLContext</span><span class="p">(</span><span class="n">sc</span><span class="p">)</span>

<span class="c"># Load a text file and convert each line to a Row.</span>
<span class="n">lines</span> <span class="o">=</span> <span class="n">sc</span><span class="o">.</span><span class="n">textFile</span><span class="p">(</span><span class="s">"/user/root/people.txt"</span><span class="p">)</span>
<span class="n">parts</span> <span class="o">=</span> <span class="n">lines</span><span class="o">.</span><span class="n">map</span><span class="p">(</span><span class="k">lambda</span> <span class="n">l</span><span class="p">:</span> <span class="n">l</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s">","</span><span class="p">))<br /><br /></span>#DataFrame is a collection of "Row" object. Create a RDD of "Row" objects and convert it into a DataFrame
<span class="n">people</span> <span class="o">=</span> <span class="n">parts</span><span class="o">.</span><span class="n">map</span><span class="p">(</span><span class="k">lambda</span> <span class="n">p</span><span class="p">:</span> <span class="n">Row</span><span class="p">(</span><span class="n">name</span><span class="o">=</span><span class="n">p</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">age</span><span class="o">=</span><span class="nb">int</span><span class="p">(</span><span class="n">p</span><span class="p">[</span><span class="mi">1</span><span class="p">])))</span>

<span class="c"># Infer the schema, and register the DataFrame as a table.</span>
<span class="n">schemaPeople</span> <span class="o">=</span> <span class="n">sqlContext</span><span class="o">.</span><span class="n">createDataFrame</span><span class="p">(</span><span class="n">people</span><span class="p">)<br /><br />#To load schema explicitly<br /></span></code></pre>
<pre class=" prettyprinted">schemaString = "name age"<code class="language-python" data-lang="python"><span class="n">people</span> <span class="o">=</span> <span class="n">parts</span><span class="o">.</span><span class="n">map</span><span class="p">(</span><span class="k">lambda</span> <span class="n">p</span><span class="p">:</span> <span class="n">Row</span><span class="p">(</span><span class="n">p</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="nb">int</span><span class="p">(</span><span class="n">p</span><span class="p">[</span><span class="mi">1</span><span class="p">]))) #Don't have to specify column names when creating the row</span></code><br />schemaPeople = sqlContext.createDataFrame(people, schema)<br /><code class="language-python" data-lang="python"><span class="p"><br /></span><span class="n">schemaPeople</span><span class="o">.</span><span class="n">registerTempTable</span><span class="p">(</span><span class="s">"people"</span><span class="p">)</span> <br /><br />#Once dataframe is registered as a temp table. It can be used a table name in SQL queries. <br /><br /><span class="n">teenagers</span> <span class="o">=</span> <span class="n">sqlContext</span><span class="o">.</span><span class="n">sql</span><span class="p">(</span><span class="s">"SELECT name FROM people WHERE age &gt;= 13 AND age &lt;= 19"</span><span class="p">)</span></code></pre>
<pre class=" prettyprinted">#Another way to create dataframe with schema</pre>
<pre class="lang-py prettyprint prettyprinted"><code><span class="pln">v </span><span class="pun">=</span><span class="pln"> sqlContext</span><span class="pun">.</span><span class="pln">createDataFrame</span><span class="pun">([</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"a"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"Alice"</span><span class="pun">,</span><span class="pln"> </span><span class="lit">34</span><span class="pun">),</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"b"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"Bob"</span><span class="pun">,</span><span class="pln"> </span><span class="lit">36</span><span class="pun">),</span><span class="pln">
  </span><span class="pun">(</span><span class="str">"c"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"Charlie"</span><span class="pun">,</span><span class="pln"> </span><span class="lit">30</span><span class="pun">),</span><span class="pln">
</span><span class="pun">],</span><span class="pln"> </span><span class="pun">[</span><span class="str">"id"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"name"</span><span class="pun">,</span><span class="pln"> </span><span class="str">"age"</span><span class="pun">]) <br /><br /></span></code></pre>
<p><strong>Load JSON Files into DataFrames</strong></p>
<pre><code class="language-python" data-lang="python"><span class="kn">from</span> <span class="nn">pyspark.sql</span> <span class="kn">import</span> <span class="n">SQLContext</span>
<span class="n">sqlContext</span> <span class="o">=</span> <span class="n">SQLContext</span><span class="p">(</span><span class="n">sc</span><span class="p">)<br /></span></code></pre>
<pre><code class="language-python" data-lang="python"><span class="n">df <span class="o">=</span> sqlContext<span class="o">.</span>read<span class="o">.</span>json<span class="p">(</span><span class="s">"/user/root/people.json"</span><span class="p">) #Uses null for missing values</span><br />df</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="n">df</span><span class="o">.</span><span class="n">printSchema</span><span class="p">()</span>
<span class="n">df</span><span class="o">.</span><span class="n">select</span><span class="p">(</span><span class="s">"name"</span><span class="p">)</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="n">df</span><span class="o">.</span><span class="n">select</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s">'name'</span><span class="p">],</span> <span class="n">df</span><span class="p">[</span><span class="s">'age'</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="n">df</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s">'age'</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mi">21</span><span class="p">)</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="n">df</span><span class="o">.</span><span class="n">groupBy</span><span class="p">(</span><span class="s">"age"</span><span class="p">)</span><span class="o">.</span><span class="n">count</span><span class="p">()</span><span class="o">.</span><span class="n">show</span><span class="p">()<br /><strong><br /><br /><span style="font-size: 10pt;">7. DataFrames - HiveContext<br /></span><br /></strong>https://spark.apache.org/docs/1.6.0/sql-programming-guide.html#hive-tables<strong><br /></strong></span></code></pre>
<pre><code class="language-python" data-lang="python"><span class="kn">from</span> <span class="nn">pyspark.sql</span> <span class="kn">import</span> <span class="n">HiveContext</span>
<span class="n">sqlContext</span> <span class="o">=</span> <span class="n">HiveContext</span><span class="p">(</span><span class="n">sc</span><span class="p">)</span>

<span class="n">sqlContext</span><span class="o">.</span><span class="n">sql</span><span class="p">(</span><span class="s">"CREATE TABLE IF NOT EXISTS src (key INT, value STRING)"</span><span class="p">)</span>
<span class="n">sqlContext</span><span class="o">.</span><span class="n">sql</span><span class="p">(</span><span class="s">"LOAD DATA INPATH '/user/root/kv1.txt' INTO TABLE src"</span><span class="p">)</span>

<span class="c"># Queries can be expressed in HiveQL.</span>
<span class="n">results</span> <span class="o">=</span> <span class="n">sqlContext</span><span class="o">.</span><span class="n">sql</span><span class="p">(</span><span class="s">"FROM src SELECT key, value"</span><span class="p">)</span><span class="o">.</span><span class="n">collect</span><span class="p">()<br /><br />#Go to Hive and you can see that the table has been created<br /><br /><span style="font-size: 10pt;"><strong>Another Trick to Writing Large Hive Tables from Spark.<br /></strong></span><br />Assumption, you have loaded and manipulated data and want the result to be loaded into a Hive Table<br /> <br />1. Create a hive table with specified "external" location and specify the delimiter (example comma)<br />2. Format the RDD into Delimited Strings (use the same delimiter that you used when creating Hive Table)<br />3. Simply do a "saveAsTextFile" and write to the location specified in "external" table<br />4. You can log into hive and run a "select * from" query to see the results. </span></code></pre>
<pre><code class="language-python" data-lang="python"><span class="p"><strong>&nbsp;</strong></span></code></pre>
<pre></pre>
</body>
</html>