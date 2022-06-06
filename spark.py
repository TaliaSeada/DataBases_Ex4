# df = sqlContext.read.option("multiline", "true").json("books.json")
# df.show()
#
# df.registerTempTable("f_table")
# start_with_f = sqlContext.sql("SELECT title, author, (2022 - year) as `number of years since published` FROM f_table WHERE author LIKE 'F%'")
# start_with_f.show()