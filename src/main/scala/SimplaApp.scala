import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.util.MLUtils
import org.jblas.DoubleMatrix

class SimpleKMeans(xK: Int, xNumIters: Int, xPoints: RDD[Array[Double]]) extends Serializable {
  // Setup
  var k = xK
  var numIterations = xNumIters
  var points = xPoints
  
  var centroids = points.take(k).toArray
  var iteration = 0
  val dims = centroids(0).length

  // Helper functions:
  // Accumulate sums and counts for each centroid
  type WeightedPoint = (DoubleMatrix, Long)
  def mergeContribs(p1: WeightedPoint, p2: WeightedPoint): WeightedPoint = {
   (p1._1.addi(p2._1), p1._2 + p2._2)
  }

  // Return the index of the centroid closest to the given point
  def findClosest(point: Array[Double]): Int = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0

    for (i <- 0 until centroids.length) {
      val distance = MLUtils.squaredDistance(point, centroids(i))
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
    }

    bestIndex
  }

  def run() {
    // Run kmeans for a fixed number of iterations
    while (iteration < numIterations) {
      // Assignment:
      // Assign each point to the closest centroid
      // Accumulating a sum and count per centroid
      val totalContribs = points.mapPartitions { points =>
        var sums = Array.fill(k)(new DoubleMatrix(dims))
        var counts = Array.fill(k)(0L)

        for (point <- points) {
          val closestIndex = findClosest(point)
          sums(closestIndex).addi(new DoubleMatrix(point))
          counts(closestIndex) += 1
        }

        val contribs = for (j <- 0 until k) yield {
          (j, (sums(j), counts(j)))
        }
        contribs.iterator
      }.reduceByKey(mergeContribs).collectAsMap()

      // Compute new centroids:
      for (j <- 0 until k) {
        val (sum, count) = totalContribs(j)
        if (count != 0) {
          val newCenter = sum.divi(count).data
          centroids(j) = newCenter
        }

      }
      iteration += 1

    }

  }

}

object SimpleApp {
  def main(args: Array[String]) {
    val conf = new SparkConf()
             .setMaster("spark://192.168.0.40:7070")
             .setAppName("Simple App")
             .setJars(List("/kmeans/target/scala-2.10/simple-project_2.10-1.0.jar"))
             .setSparkHome("/software/spark-0.9.1")
             .set("spark.executor.memory", "1g")
             .set("spark.cores.max", "2")
    val sc = new SparkContext(conf)

    val filePath = "hdfs://192.168.0.40:54310/josh/foo.csv"
    val points = sc.textFile(filePath).map( _.split(',').map(_.toDouble)).cache()
    val model = new SimpleKMeans(2, 10, points)
    model.run

  }
}
