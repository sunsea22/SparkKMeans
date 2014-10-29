/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.regression.LabeledPoint
import org.jblas.DoubleMatrix

class SimpleSGD(xNumIters: Int, xPoints: RDD[LabeledPoint]) extends Serializable {
  // Setup
  var points = xPoints
  var numIters = xNumIters
  val lambda = .1
  var alpha = .1
 
  var first = points.take(1).toArray
  val dims = first(0).features.length
  var params = new DoubleMatrix(dims)
  val absolute_start = System.currentTimeMillis
 
  var iteration = 0
  var hack = System.currentTimeMillis
  var totalIterTime = hack - hack

  // Helper functions:
  // Accumulate sums and counts for each centroid
  type WeightedPoint = (DoubleMatrix, Int)
  def mergeContribs(p1: WeightedPoint, p2: WeightedPoint): WeightedPoint = {
   (p1._1.addi(p2._1), p1._2 + p2._2)
  }


  def gradient(point: LabeledPoint): DoubleMatrix = {
    val features = new DoubleMatrix(point.features)
    val flag = point.label * params.dot(features)
    if (flag > 1) {
      params.mul(lambda)
    } else {
      params.mul(lambda).sub(features.mul(point.label))
    }
  }

  def run() {
    // Run kmeans for a fixed number of iterations
    while (iteration < numIters) {
      val start = System.currentTimeMillis
      
      val (sum,count) = points.mapPartitions { points =>
        for (point <- points) {
          val update = gradient(point).mul(alpha)
          params.subi(update)
        }

        val foo = for (j <- 0 until 1) yield {
          (params, 1)
        }

        foo.iterator
      }.reduce(mergeContribs)

      // Compute new parameters
      params = sum.divi(count)
      
      iteration += 1
      alpha = .95 * alpha
      val end  = System.currentTimeMillis
      val elapsed = end - start
      totalIterTime += elapsed
    }
    val absolute_end = System.currentTimeMillis
    println("Total time after loading: ")
    println(absolute_end - absolute_start)
    println("Average iteration time")
    println(totalIterTime / numIters) 
    println("Params:")
    println(params.data.mkString(","))
  }

}

object SimpleApp {
  def main(args: Array[String]) {
    val conf = new SparkConf()
             .setMaster("spark://192.168.0.40:7070")
             .setAppName("Simple App")
             .setJars(List("/SparkSGD/target/scala-2.10/simple-project_2.10-1.0.jar"))
             .setSparkHome("/software/spark-0.9.1")
             .set("spark.executor.memory", "65g")
             .set("spark.cores.max", "1")
    val sc = new SparkContext(conf)

    val filePath = "hdfs://192.168.0.40:54310/josh/foolabeled.csv"
    val points = sc.textFile(filePath).map{ line => 
      val parts = line.split(',')
      LabeledPoint(parts.last.toDouble, parts.init.map(x => x.toDouble).toArray)
    }.cache()
    val model = new SimpleSGD(10, points)
    model.run

  }
}
