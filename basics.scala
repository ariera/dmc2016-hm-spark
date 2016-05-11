// bin/spark-shell --packages com.databricks:spark-csv_2.10:1.3.0
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType}
import org.apache.spark.mllib.regression.{LabeledPoint}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val sqlContext = new SQLContext(sc)

val customSchema = StructType(
  Array(
    StructField("customerid" , StringType, true),
    StructField("size_bought_times" , FloatType, true),
    StructField("size_returned_ratio" , FloatType, true),
    StructField("size_returned_times" , FloatType, true),
    StructField("color_bought_times" , FloatType, true),
    StructField("color_returned_ratio" , FloatType, true),
    StructField("color_returned_times" , FloatType, true),
    StructField("customer_return_ratio" , FloatType, true),
    StructField("customer_sum_quantities" , FloatType, true),
    StructField("customer_sum_returns" , FloatType, true),
    StructField("colorcode" , StringType, true),
    StructField("deviceid" , StringType, true),
    StructField("day in month" , StringType, true),
    StructField("month_of_year" , StringType, true),
    StructField("day_of_week" , StringType, true),
    StructField("quarter" , StringType, true),
    StructField("orderid" , StringType, true),
    StructField("articleid" , StringType, true),
    StructField("sizecode" , StringType, true),
    StructField("productgroup" , StringType, true),
    StructField("quantity" , FloatType, true),
    StructField("price" , FloatType, true),
    StructField("rrp" , FloatType, true),
    StructField("voucherid" , StringType, true),
    StructField("voucheramount" , FloatType, true),
    StructField("paymentmethod" , StringType, true),
    StructField("orderdate" , StringType, true),
    StructField("price_per_item" , FloatType, true),
    StructField("price_to_rrp_ratio" , FloatType, true),
    StructField("usual_price_ratio" , FloatType, true),
    StructField("color_ral_group" , StringType, true),
    StructField("has_voucher" , StringType, true),
    StructField("article_average_price" , FloatType, true),
    StructField("article_cheapest_price" , FloatType, true),
    StructField("article_most_expensive_price" , FloatType, true),
    StructField("article_number_of_different_prices" , FloatType, true),
    StructField("total_order_price" , FloatType, true),
    StructField("different_sizes" , StringType, true),
    StructField("sizes" , StringType, true),
    StructField("different_colors" , StringType, true),
    StructField("colors" , StringType, true),
    StructField("NewProductGroup" , StringType, true),
    StructField("NewSizeCode" , StringType, true),
    StructField("new_paymentmethod" , StringType, true),
    StructField("year_and_month" , StringType, true),
    StructField("id" , StringType, true),
    StructField("returnquantity" , FloatType, true)
    )
  )


    StructField("orderid"       , StringType , true),
    StructField("articleid"     , StringType , true),
    StructField("colorcode"     , StringType , true),
    StructField("sizecode"      , StringType , true),
    StructField("productgroup"  , StringType , true),
    StructField("quantity"      , FloatType, true),
    StructField("price"         , FloatType  , true),
    StructField("rrp"           , FloatType  , true),
    StructField("voucherid"     , StringType , true),
    StructField("voucheramount" , FloatType  , true),
    StructField("customerid"    , StringType , true),
    StructField("deviceid"      , StringType , true),
    StructField("paymentmethod" , StringType , true),
    StructField("returnquantity", FloatType, true),
    StructField("id"            , StringType , true),
    StructField("orderdate"     , StringType , true)
  )
)

val df = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(customSchema) .load("dm2_train_sample.csv")


val l = List("orderid", "articleid", "colorcode", "sizecode",
    "productgroup", "voucherid", "customerid", "deviceid", "paymentmethod",
    "id", "orderdate"
    )

val df2= df.select("orderid", "articleid", "colorcode", "sizecode",
    "productgroup", "voucherid", "customerid", "deviceid", "paymentmethod",
    "id", "orderdate"
    )

val i = 0;
val categoricalFeaturesInfo = Map[Int, Int]

l.foreach(
    val mapToNumeric = df2.select(_).distinct.map(row => row.getString(i)).zipWithIndex.collect.toMap

    catagoricalFeaturesInfo(i -> mapToNumeric.size)

    val dataSet = df.map(row =>
        val numericOrderid = mapToNumeric(row.get(i))
    )
    i = i + 1;
    )


/home/axel/Skrivbord/allAttributes/all_features_splited_v3/dm2_train_and_test_v3/dm2_train_known_customer_v3.csv

Test known

/home/axel/Skrivbord/allAttributes/all_features_splited_v3/dm2_train_and_test_v3/dm2_test_kwown_customers_v3.csv


