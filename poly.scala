./bin/spark-shell --packages com.databricks:spark-csv_2.11:1.4.0


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType}
import org.apache.spark.mllib.regression.{LabeledPoint}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val knownCustTrainSchema = StructType(
  Array(
    StructField("voucherid" , FloatType, true),
    StructField("customerid" , FloatType, true),
    StructField("colorcode" , FloatType, true),
    StructField("deviceid" , FloatType, true),
    StructField("day in month" , FloatType, true),
    StructField("month_of_year" , FloatType, true),
    StructField("day_of_week" , FloatType, true),
    StructField("quarter" , FloatType, true),
    StructField("paymentmethod" , FloatType, true),
    StructField("has_voucher" , FloatType, true),
    StructField("NewProductGroup" , FloatType, true),
    StructField("NewSizeCode" , FloatType, true),
    StructField("new_paymentmethod" , FloatType, true),
    StructField("sizecode" , FloatType, true),
    StructField("orderid" , FloatType, true),
    StructField("articleid" , FloatType, true),
    StructField("productgroup" , FloatType, true),
    StructField("sizes" , FloatType, true),
    StructField("colors" , FloatType, true),
    StructField("year_and_month" , FloatType, true),
    StructField("orderdate" , FloatType, true),
    StructField("quantity" , FloatType, true),
    StructField("price" , FloatType, true),
    StructField("rrp" , FloatType, true),
    StructField("voucheramount" , FloatType, true),
    StructField("price_per_item" , FloatType, true),
    StructField("price_to_rrp_ratio" , FloatType, true),
    StructField("usual_price_ratio" , FloatType, true),
    StructField("color_ral_group" , FloatType, true),
    StructField("article_average_price" , FloatType, true),
    StructField("article_cheapest_price" , FloatType, true),
    StructField("article_most_expensive_price" , FloatType, true),
    StructField("article_number_of_different_prices" , FloatType, true),
    StructField("total_order_price" , FloatType, true),
    StructField("different_sizes" , FloatType, true),
    StructField("different_colors" , FloatType, true),
    StructField("color_returned_times" , FloatType, true),
    StructField("color_bought_times" , FloatType, true),
    StructField("color_returned_ratio" , FloatType, true),
    StructField("size_returned_times" , FloatType, true),
    StructField("size_bought_times" , FloatType, true),
    StructField("size_returned_ratio" , FloatType, true),
    StructField("customer_sum_quantities" , FloatType, true),
    StructField("customer_sum_returns" , FloatType, true),
    StructField("customer_return_ratio" , FloatType, true),
    StructField("returnquantity" , FloatType, true),
    StructField("id" , FloatType, true)
    )
  )

val loadTrainKnown = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(knownCustTrainSchema) .load("/home/axel/Skrivbord/allAttributes/all_features_splited_v3/dm2_train_and_test_v3/dm2_train_known_customer_v3.csv")

val trainKnown = loadTrainKnown.select(
    "voucherid" , 
    "colorcode" , 
    "deviceid" , 
    "day in month" , 
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

val knownCustTestSchema = StructType(
  Array(
    StructField("customerid" , FloatType, true),
    StructField("size_bought_times" , FloatType, true),
    StructField("size_returned_ratio" , FloatType, true),
    StructField("size_returned_times" , FloatType, true),
    StructField("color_bought_times" , FloatType, true),
    StructField("color_returned_ratio" , FloatType, true),
    StructField("color_returned_times" , FloatType, true),
    StructField("customer_return_ratio" , FloatType, true),
    StructField("customer_sum_quantities" , FloatType, true),
    StructField("customer_sum_returns" , FloatType, true),
    StructField("voucherid" , FloatType, true),
    StructField("colorcode" , FloatType, true),
    StructField("deviceid" , FloatType, true),
    StructField("day in month" , FloatType, true),
    StructField("month_of_year" , FloatType, true),
    StructField("day_of_week" , FloatType, true),
    StructField("quarter" , FloatType, true),
    StructField("paymentmethod" , FloatType, true),
    StructField("has_voucher" , FloatType, true),
    StructField("NewProductGroup" , FloatType, true),
    StructField("NewSizeCode" , FloatType, true),
    StructField("new_paymentmethod" , FloatType, true),
    StructField("sizecode" , FloatType, true),
    StructField("orderid" , FloatType, true),
    StructField("articleid" , FloatType, true),
    StructField("productgroup" , FloatType, true),
    StructField("sizes" , FloatType, true),
    StructField("orderdate" , FloatType, true),
    StructField("colors" , FloatType, true),
    StructField("year_and_month" , FloatType, true),
    StructField("quantity" , FloatType, true),
    StructField("price" , FloatType, true),
    StructField("rrp" , FloatType, true),
    StructField("voucheramount" , FloatType, true),
    StructField("price_per_item" , FloatType, true),
    StructField("price_to_rrp_ratio" , FloatType, true),
    StructField("usual_price_ratio" , FloatType, true),
    StructField("color_ral_group" , FloatType, true),
    StructField("article_average_price" , FloatType, true),
    StructField("article_cheapest_price" , FloatType, true),
    StructField("article_most_expensive_price" , FloatType, true),
    StructField("article_number_of_different_prices" , FloatType, true),
    StructField("total_order_price" , FloatType, true),
    StructField("different_sizes" , FloatType, true),
    StructField("different_colors" , FloatType, true),
    StructField("returnquantity" , FloatType, true),
    StructField("id" , FloatType, true)
    )
  )

val loadTestKnown = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(knownCustTestSchema) .load("/home/axel/Skrivbord/allAttributesNumeric/dm2_train_and_test_v3.numeric/dm2_test_kwown_customers_v3.numeric.csv")



val newCustTrainSchema = StructType(
    Array(
    StructField("voucherid" , FloatType, true),
    StructField("colorcode" , FloatType, true),
    StructField("deviceid" , FloatType, true),
    StructField("day in month" , FloatType, true),
    StructField("month_of_year" , FloatType, true),
    StructField("day_of_week" , FloatType, true),
    StructField("quarter" , FloatType, true),
    StructField("paymentmethod" , FloatType, true),
    StructField("has_voucher" , FloatType, true),
    StructField("NewProductGroup" , FloatType, true),
    StructField("NewSizeCode" , FloatType, true),
    StructField("new_paymentmethod" , FloatType, true),
    StructField("sizecode" , FloatType, true),
    StructField("orderid" , FloatType, true),
    StructField("articleid" , FloatType, true),
    StructField("productgroup" , FloatType, true),
    StructField("sizes" , FloatType, true),
    StructField("colors" , FloatType, true),
    StructField("year_and_month" , FloatType, true),
    StructField("orderdate" , FloatType, true),
    StructField("quantity" , FloatType, true),
    StructField("price" , FloatType, true),
    StructField("rrp" , FloatType, true),
    StructField("voucheramount" , FloatType, true),
    StructField("price_per_item" , FloatType, true),
    StructField("price_to_rrp_ratio" , FloatType, true),
    StructField("usual_price_ratio" , FloatType, true),
    StructField("color_ral_group" , FloatType, true),
    StructField("article_average_price" , FloatType, true),
    StructField("article_cheapest_price" , FloatType, true),
    StructField("article_most_expensive_price" , FloatType, true),
    StructField("article_number_of_different_prices" , FloatType, true),
    StructField("total_order_price" , FloatType, true),
    StructField("different_sizes" , FloatType, true),
    StructField("different_colors" , FloatType, true),
    StructField("returnquantity" , FloatType, true),
    StructField("id" , FloatType, true)
        )
    )

val loadTrainNew = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(newCustTrainSchema) .load("/home/axel/Skrivbord/allAttributesNumeric/dm2_train_and_test_v3.numeric/dm2_train_new_customer_v3.numeric.csv")

val newCustTestSchema = StructType(
    Array(
    StructField("voucherid" , FloatType, true),
    StructField("colorcode" , FloatType, true),
    StructField("deviceid" , FloatType, true),
    StructField("day in month" , FloatType, true),
    StructField("month_of_year" , FloatType, true),
    StructField("day_of_week" , FloatType, true),
    StructField("quarter" , FloatType, true),
    StructField("paymentmethod" , FloatType, true),
    StructField("has_voucher" , FloatType, true),
    StructField("NewProductGroup" , FloatType, true),
    StructField("NewSizeCode" , FloatType, true),
    StructField("new_paymentmethod" , FloatType, true),
    StructField("sizecode" , FloatType, true),
    StructField("orderid" , FloatType, true),
    StructField("articleid" , FloatType, true),
    StructField("productgroup" , FloatType, true),
    StructField("sizes" , FloatType, true),
    StructField("orderdate" , FloatType, true),
    StructField("colors" , FloatType, true),
    StructField("year_and_month" , FloatType, true),
    StructField("quantity" , FloatType, true),
    StructField("price" , FloatType, true),
    StructField("rrp" , FloatType, true),
    StructField("voucheramount" , FloatType, true),
    StructField("price_per_item" , FloatType, true),
    StructField("price_to_rrp_ratio" , FloatType, true),
    StructField("usual_price_ratio" , FloatType, true),
    StructField("color_ral_group" , FloatType, true),
    StructField("article_average_price" , FloatType, true),
    StructField("article_cheapest_price" , FloatType, true),
    StructField("article_most_expensive_price" , FloatType, true),
    StructField("article_number_of_different_prices" , FloatType, true),
    StructField("total_order_price" , FloatType, true),
    StructField("different_sizes" , FloatType, true),
    StructField("different_colors" , FloatType, true),
    StructField("returnquantity" , FloatType, true),
    StructField("id" , FloatType, true)
        )
    )

val loadTestKnown = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(newCustTestSchema) .load("/home/axel/Skrivbord/allAttributesNumeric/dm2_train_and_test_v3.numeric/dm2_test_new_customers_v3.numeric.csv")
