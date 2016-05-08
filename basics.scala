// bin/spark-shell --packages com.databricks:spark-csv_2.10:1.3.0
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType}
import org.apache.spark.mllib.regression.{LabeledPoint}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val sqlContext = new SQLContext(sc)

val customSchema = StructType(
  Array(
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



