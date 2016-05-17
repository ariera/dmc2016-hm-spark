
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils


val labeledKnownTr = MLUtils.loadLibSVMFile(sc, "labeledKnownTr10")

val labeledKnownTe = MLUtils.loadLibSVMFile(sc, "labeledKnownTe10")

//Key starts at 0 (). Value represents no of different features
//Will work with a proper encoding
val knownCategoricalFeaturesInfo = Map[Int, Int](
    // //1 -> 267, 
    // //2 -> 492, 
    //0 -> 341
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
    val numClasses = 2
    val numTrees = 500
    val featureSubsetStrategy = "sqrt" 
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 100

val model = RandomForest.trainClassifier(labeledKnownTr, numClasses, knownCategoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)



val labelAndPreds = labeledKnownTe.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / labeledKnownTe.count()
println("Test Error = " + testErr)
println("Learned classification forest model:\n" + model.toDebugString)


val model = RandomForestModel.load(sc, "treeEnsambleModel")