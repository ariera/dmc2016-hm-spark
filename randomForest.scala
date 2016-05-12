
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils


//Key starts at 0 (). Value represents no of different features
//Will work with a proper encoding
val knownCategoricalFeaturesInfo = Map[Int, Int](
    // //1 -> 267, 
    // //2 -> 492, 
    0 -> 5, 
    1 -> 31, 
    3 -> 12, 
    4 -> 7, 
    5 -> 9, 
    6 -> 2
    // 7 -> 2, 
    // 8 -> 3, 
    // 9 -> 3, 
    //10 -> 2, 
    //11 -> 29
    // //15 -> 3409, 
    // 16 -> 14, 
    // //17 -> 153, 
    // //18 -> 5405, 
    // 19 -> 18, 
    // 28 -> 5, 
    // 34 -> 5, 
    // 35 -> 8
    )
    val numClasses = 6
    val numTrees = 500
    val featureSubsetStrategy = "auto" 
    val impurity = "gini"
    val maxDepth = 6
    val maxBins = 32

val model = RandomForest.trainClassifier(labeledKnownTr, numClasses, knownCategoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)




val model = RandomForestModel.load(sc, "treeEnsambleModel")

val labelAndPreds = labeledKnownTe.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / labeledKnownTe.count()
println("Test Error = " + testErr)
println("Learned classification forest model:\n" + model.toDebugString)