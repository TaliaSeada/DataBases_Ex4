# df = sqlContext.read.option("multiline", "true").json("books.json")
# df.show()

# # SQL
# df.registerTempTable("f_table")
# start_with_f = sqlContext.sql("SELECT title, author, (2022 - year) as `number of years since published` FROM f_table WHERE author LIKE 'F%'")
# start_with_f.show()

# # Spark
# start_F = start_F.select("title", "author", "year")
# start_F = start_F.withColumn("number of years since published", 2022 - start_F.year)
# start_F = start_F.filter("author like 'F%'")
# final = start_F.select("title", "author", "number of years since published")
# final.show()