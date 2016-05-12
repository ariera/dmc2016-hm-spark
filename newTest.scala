val newCustTestSchema = StructType(
    Array(
    StructField("voucherid" , StringType, true),
    StructField("colorcode" , StringType, true),
    StructField("deviceid" , StringType, true),
    StructField("day_in_month" , StringType, true),
    StructField("month_of_year" , StringType, true),
    StructField("day_of_week" , StringType, true),
    StructField("quarter" , StringType, true),
    StructField("paymentmethod" , StringType, true),
    StructField("has_voucher" , StringType, true),
    StructField("NewProductGroup" , StringType, true),
    StructField("NewSizeCode" , StringType, true),
    StructField("new_paymentmethod" , StringType, true),
    StructField("sizecode" , StringType, true),
    StructField("orderid" , StringType, true),
    StructField("articleid" , StringType, true),
    StructField("productgroup" , StringType, true),
    StructField("sizes" , StringType, true),
    StructField("orderdate" , StringType, true),
    StructField("colors" , StringType, true),
    StructField("year_and_month" , DoubleType, true),
    StructField("quantity" , DoubleType, true),
    StructField("price" , DoubleType, true),
    StructField("rrp" , DoubleType, true),
    StructField("voucheramount" , DoubleType, true),
    StructField("price_per_item" , DoubleType, true),
    StructField("price_to_rrp_ratio" , DoubleType, true),
    StructField("usual_price_ratio" , DoubleType, true),
    StructField("color_ral_group" , StringType, true),
    StructField("article_average_price" , DoubleType, true),
    StructField("article_cheapest_price" , DoubleType, true),
    StructField("article_most_expensive_price" , DoubleType, true),
    StructField("article_number_of_different_prices" , DoubleType, true),
    StructField("total_order_price" , DoubleType, true),
    StructField("different_sizes" , StringType, true),
    StructField("different_colors" , StringType, true),
    StructField("returnquantity" , DoubleType, true),
    StructField("id" , DoubleType, true)
        )
    )
val newTestLoad = sqlContext.read .format("com.databricks.spark.csv") .option("header", "true") .option("delimiter", ";") .schema(newCustTestSchema) .load("/home/axel/Skrivbord/allAttributes/all_features_splited_v3/dm2_train_and_test_v3/dm2_test_new_customers_v3.csv")
val newTest = newTestLoad.select(
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
    "id" ,
    "returnquantity" 
    )

val newAssembler = new VectorAssembler().setInputCols(Array(
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
    "id"
    )
  ).setOutputCol("features")


val newTr = knownAssembler.transform(knownTrain).select("features", "returnquantity")

val newTe = knownAssembler.transform(knownTest).select("features", "returnquantity")


val labeledNewTr = knownTr.map(row => LabeledPoint(row.getDouble(1), row(0).asInstanceOf[Vector]))

val labeledNewTe = knownTe.map(row => LabeledPoint(row.getDouble(1), row(0).asInstanceOf[Vector]))

val newCategoricalFeaturesInfo = Map[Int, Int](
    1 -> 
    2 -> 
    3 -> 5
    4 -> 31
    5 -> 12
    6 -> 7
    7 -> 4
    8 -> 9
    9 -> 2
    10 -> 3
    11 -> 3
    12 -> 2
    13 -> 
    15 -> 
    16 -> 
    17 -> 
    18 -> 
    19 -> 
    28 -> 
    )




// NEW CUSTOMERS

