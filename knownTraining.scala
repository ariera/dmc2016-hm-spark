//Copy all this and paste:
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DoubleType}
import org.apache.spark.mllib.regression.{LabeledPoint}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer


val knownCustTrainSchema = StructType(
  Array(
    StructField("colorcode" , StringType, true),
    StructField("deviceid" , StringType, true),
    StructField("day_in_month" , StringType, true),
    StructField("month_of_year" , StringType, true),
    StructField("day_of_week" , StringType, true),
    StructField("quarter" , StringType, true),
    StructField("orderid" , StringType, true),
    StructField("articleid" , StringType, true),
    StructField("sizecode" , StringType, true),
    StructField("productgroup" , StringType, true),
    StructField("quantity" , DoubleType, true),
    StructField("price" , DoubleType, true),
    StructField("rrp" , DoubleType, true),
    StructField("voucheramount" , DoubleType, true),
    StructField("voucherid" , StringType, true),
    StructField("customerid" , StringType, true),
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
    StructField("different_sizes" , DoubleType, true),
    StructField("sizes" , StringType, true),
    StructField("different_colors" , DoubleType, true),
    StructField("colors" , StringType, true),
    StructField("color_returned_times" , DoubleType, true),
    StructField("color_bought_times" , DoubleType, true),
    StructField("color_returned_ratio" , DoubleType, true),
    StructField("size_returned_times" , DoubleType, true),
    StructField("size_bought_times" , DoubleType, true),
    StructField("size_returned_ratio" , DoubleType, true),
    StructField("customer_sum_quantities" , DoubleType, true),
    StructField("customer_sum_returns" , DoubleType, true),
    StructField("customer_return_ratio" , DoubleType, true),
    StructField("NewProductGroup" , StringType, true),
    StructField("NewSizeCode" , StringType, true),
    StructField("new_paymentmethod" , StringType, true),
    StructField("year_and_month" , StringType, true),
    StructField("returnquantity" , DoubleType, true),
    StructField("id" , DoubleType, true)
    )
  )
val knownTrainLoad = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(knownCustTrainSchema) .load("/home/axel/Skrivbord/allAttributes/all_features_splited_v3/dm2_train_and_test_v3/dm2_train_known_customer_v3.csv")
// .select to get the same order of the attributes (this is done for all four datasets)
val knownTrain = knownTrainLoad.select(
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


val voucheridIndexer = new StringIndexer().setInputCol("voucherid" ).setOutputCol("voucheridIndexed");
val customeridIndexer = new StringIndexer().setInputCol("customerid").setOutputCol("customeridIndexed");
val colorcodeIndexer = new StringIndexer().setInputCol("colorcode").setOutputCol("colorcodeIndexed");
val deviceidIndexer = new StringIndexer().setInputCol("deviceid").setOutputCol("deviceidIndexed");
val day_in_monthIndexer = new StringIndexer().setInputCol("day_in_month").setOutputCol("day_in_monthIndexed");
val month_of_yearIndexer = new StringIndexer().setInputCol("month_of_year").setOutputCol("month_of_yearIndexed");
val day_of_weekIndexer = new StringIndexer().setInputCol("day_of_week").setOutputCol("day_of_weekIndexed");
val quarterIndexer = new StringIndexer().setInputCol("quarter").setOutputCol("quarterIndexed");
val paymentmethodIndexer = new StringIndexer().setInputCol("paymentmethod").setOutputCol("paymentmethodIndexed");
val has_voucherIndexer = new StringIndexer().setInputCol("has_voucher").setOutputCol("has_voucherIndexed");
val NewProductGroupIndexer = new StringIndexer().setInputCol("NewProductGroup").setOutputCol("NewProductGroupIndexed");
val NewSizeCodeIndexer = new StringIndexer().setInputCol("NewSizeCode").setOutputCol("NewSizeCodeIndexed");
val new_paymentmethodIndexer = new StringIndexer().setInputCol("new_paymentmethod").setOutputCol("new_paymentmethodIndexed");
val sizecodeIndexer = new StringIndexer().setInputCol("sizecode").setOutputCol("sizecodeIndexed");
val orderidIndexer = new StringIndexer().setInputCol("orderid").setOutputCol("orderidIndexed");
val articleidIndexer = new StringIndexer().setInputCol("articleid").setOutputCol("articleidIndexed");
val productgroupIndexer = new StringIndexer().setInputCol("productgroup").setOutputCol("productgroupIndexed");
val sizesIndexer = new StringIndexer().setInputCol("sizes").setOutputCol("sizesIndexed");
val colorsIndexer = new StringIndexer().setInputCol("colors").setOutputCol("colorsIndexed");
val year_and_monthIndexer = new StringIndexer().setInputCol("year_and_month").setOutputCol("year_and_monthIndexed");
val orderdateIndexer = new StringIndexer().setInputCol("orderdate" ).setOutputCol("orderdateIndexed");
val color_ral_groupIndexer = new StringIndexer().setInputCol("color_ral_group" ).setOutputCol("color_ral_groupIndexed")

val voucheridIndexed = voucheridIndexer.fit(knownTrain).transform(knownTrain)
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
val knownTrain2 = color_ral_groupIndexer.fit(orderdateIndexed).transform(orderdateIndexed)



//Creates Vector of every Row and outputs as "feature"
val knownAssembler = new VectorAssembler().setInputCols(Array(
    // "voucheridIndexed" , 
    // "colorcodeIndexed" , 
     "deviceidIndexed" , 
     "day_in_monthIndexed" , 
     "month_of_yearIndexed" , 
     "day_of_weekIndexed" , 
     "quarterIndexed" , 
     "paymentmethodIndexed" , 
     "has_voucherIndexed" , 
    // "NewProductGroupIndexed" , 
    // "NewSizeCodeIndexed" , 
    // "new_paymentmethodIndexed" , 
    // "sizecodeIndexed" , 
    // "orderidIndexed" , 
    // "articleidIndexed" , 
    // "productgroupIndexed" , 
     //"sizesIndexed" , 
     //"colorsIndexed" , 
    // "year_and_monthIndexed" , 
    // "orderdateIndexed" , 
     "quantity" , 
     "price" , 
     "rrp" , 
    // "voucheramount" , 
     "price_per_item" , 
    // "price_to_rrp_ratio" , 
     "usual_price_ratio" , 
    // "color_ral_groupIndexed" , 
    // "article_average_price" , 
    // "article_cheapest_price" , 
     "article_most_expensive_price" , 
    // "article_number_of_different_prices" , 
    // "total_order_price" , 
    // "different_sizes" , 
    // "different_colors"  ,
    // "customeridIndexde" , 
     "color_returned_times" , 
     "color_bought_times" , 
     "color_returned_ratio" , 
     "size_returned_times" , 
     "size_bought_times" , 
     "size_returned_ratio" , 
     "customer_sum_quantities" , 
     "customer_sum_returns" , 
     "customer_return_ratio" ,
     "id"
    )
).setOutputCol("features")

val knownTr = knownAssembler.transform(knownTrain2).select("features", "returnquantity")

val labeledKnownTr = knownTr.map(row => LabeledPoint(row.getDouble(1), row(0).asInstanceOf[Vector]))