//from bash in sparkfolder

./bin/spark-shell --packages com.databricks:spark-csv_2.11:1.4.0 

//Copy all this and paste:
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DoubleType}
import org.apache.spark.mllib.regression.{LabeledPoint}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

val knownCustTestSchema = StructType(
  Array(
    StructField("customerid", StringType, true),
    StructField("size_bought_times", DoubleType, true),
    StructField("size_returned_ratio", DoubleType, true),
    StructField("size_returned_times", DoubleType, true),
    StructField("color_bought_times", DoubleType, true),
    StructField("color_returned_ratio", DoubleType, true),
    StructField("color_returned_times", DoubleType, true),
    StructField("customer_return_ratio", DoubleType, true),
    StructField("customer_sum_quantities", DoubleType, true),
    StructField("customer_sum_returns", DoubleType, true),
    StructField("colorcode", StringType, true),
    StructField("deviceid", StringType, true),
    StructField("day_in_month", StringType, true),
    StructField("month_of_year", StringType, true),
    StructField("day_of_week", StringType, true),
    StructField("quarter", StringType, true),
    StructField("orderid", StringType, true),
    StructField("articleid", StringType, true),
    StructField("sizecode", StringType, true),
    StructField("productgroup", StringType, true),
    StructField("quantity", DoubleType, true),
    StructField("price", DoubleType, true),
    StructField("rrp", DoubleType, true),
    StructField("voucherid" , StringType, true),
    StructField("voucheramount" , DoubleType, true),
    StructField("paymentmethod" , StringType, true),
    StructField("orderdate" , StringType, true),
    StructField("price_per_item" , DoubleType, true),
    StructField("price_to_rrp_ratio" , DoubleType, true),
    StructField("usual_price_ratio" , DoubleType, true),
    StructField("color_ral_group" , StringType, true),
    StructField("has_voucher" , StringType, true),
    StructField("article_average_price" , DoubleType, true),
    StructField("article_cheapest_price" , DoubleType, true),
    StructField("article_most_expensive_price" , DoubleType, true),
    StructField("article_number_of_different_prices" , DoubleType, true),
    StructField("total_order_price" , DoubleType, true),
    StructField("different_sizes" , StringType, true),
    StructField("sizes" , StringType, true),
    StructField("different_colors" , StringType, true),
    StructField("colors" , StringType, true),
    StructField("NewProductGroup" ,StringType, true),
    StructField("NewSizeCode" , StringType, true),
    StructField("new_paymentmethod" ,StringType, true),
    StructField("year_and_month" ,StringType, true),
    StructField("id" , StringType, true),
    StructField("returnquantity", DoubleType, true)
    )
  )
val knownTestLoad = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(knownCustTestSchema) .load("path/to/textFile")

val knownTest = knownTestLoad.select(
    "voucherid" , 
    "colorcode" , 
    "deviceid" , 
    "day_in_month" , 
    "month_of_year" , 
    "day_of_week" , 
    "quarter" , 
    "paymentmethod" , 
    "has_voucher" , 
    "NewProductGroup" , 
    "NewSizeCode" , 
    "new_paymentmethod" , 
    "sizecode" , 
    "orderid" , 
    "articleid" , 
    "productgroup" , 
    "sizes" , 
    "colors" , 
    "year_and_month" , 
    "orderdate" , 
    "quantity" , 
    "price" , 
    "rrp" , 
    "voucheramount" , 
    "price_per_item" , 
    "price_to_rrp_ratio" , 
    "usual_price_ratio" , 
    "color_ral_group" , 
    "article_average_price" , 
    "article_cheapest_price" , 
    "article_most_expensive_price" , 
    "article_number_of_different_prices" , 
    "total_order_price" , 
    "different_sizes" , 
    "different_colors" , 
    "customerid" , 
    "color_returned_times" , 
    "color_bought_times" , 
    "color_returned_ratio" , 
    "size_returned_times" , 
    "size_bought_times" , 
    "size_returned_ratio" , 
    "customer_sum_quantities" , 
    "customer_sum_returns" , 
    "customer_return_ratio" , 
    "id" ,
    "returnquantity" 
    )

//Use same indexer as knownTrain
val voucheridIndexed = voucheridIndexer.fit(knownTest).transform(knownTest)
val customeridIndexed = customeridIndexer.fit(voucheridIndexed).transform(voucheridIndexed)
val colorcodeIndexed = colorcodeIndexer.fit(customeridIndexed).transform(customeridIndexed)
val deviceidIndexed = deviceidIndexer.fit(colorcodeIndexed).transform(colorcodeIndexed)
val day_in_monthIndexed = day_in_monthIndexer.fit(deviceidIndexed).transform(deviceidIndexed)
val month_of_yearIndexed = month_of_yearIndexer.fit(day_in_monthIndexed).transform(day_in_monthIndexed)
val day_of_weekIndexed = day_of_weekIndexer.fit(month_of_yearIndexed).transform(month_of_yearIndexed)
val quarterIndexed = quarterIndexer.fit(day_of_weekIndexed).transform(day_of_weekIndexed)
val paymentmethodIndexed = paymentmethodIndexer.fit(quarterIndexed).transform(quarterIndexed)
val has_voucherIndexed = has_voucherIndexer.fit(paymentmethodIndexed).transform(paymentmethodIndexed)
val NewProductGroupIndexed = NewProductGroupIndexer.fit(has_voucherIndexed).transform(has_voucherIndexed)
val NewSizeCodeIndexed = NewSizeCodeIndexer.fit(NewProductGroupIndexed).transform(NewProductGroupIndexed)
val new_paymentmethodIndexed = new_paymentmethodIndexer.fit(NewSizeCodeIndexed).transform(NewSizeCodeIndexed)
val sizecodeIndexed = sizecodeIndexer.fit(new_paymentmethodIndexed).transform(new_paymentmethodIndexed)
val orderidIndexed = orderidIndexer.fit(sizecodeIndexed).transform(sizecodeIndexed)
val articleidIndexed = articleidIndexer.fit(orderidIndexed).transform(orderidIndexed)
val productgroupIndexed = productgroupIndexer.fit(articleidIndexed).transform(articleidIndexed)
val sizesIndexed = sizesIndexer.fit(productgroupIndexed).transform(productgroupIndexed)
val colorsIndexed = colorsIndexer.fit(sizesIndexed).transform(sizesIndexed)
val year_and_monthIndexed = year_and_monthIndexer.fit(colorsIndexed).transform(colorsIndexed)
val orderdateIndexed = orderdateIndexer.fit(year_and_monthIndexed).transform(year_and_monthIndexed)
val knownTest2 = color_ral_groupIndexer.fit(orderdateIndexed).transform(orderdateIndexed)

 //Choose the attributes that ChiSq selected, in this case 6
  val knownAssembler = new VectorAssembler().setInputCols(Array(
        "quantity" , 
        "price" , 
        "total_order_price" , 
        "color_returned_times" , 
        "color_returned_ratio" , 
        "size_returned_ratio" 
      )
  ).setOutputCol("features")

val knownTe = knownAssembler.transform(knownTest2).select("features", "returnquantity")

val labeledKnownTe = knownTe.map(row => LabeledPoint(row.getDouble(1), row(0).asInstanceOf[Vector]))

labeledKnownTe.saveAsTextFile(sc, "labeledKnownTe")
