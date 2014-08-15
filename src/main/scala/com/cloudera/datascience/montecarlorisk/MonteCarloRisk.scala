/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.cloudera.datascience.montecarlorisk

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.serializer.{KryoSerializer, KryoRegistrator}
import com.esotericsoftware.kryo.Kryo
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.distribution.MultivariateNormalDistribution

import scala.io.Source
import java.io.PrintWriter

case class Instrument(factorWeights: Array[Double], minValue: Double = 0,
  maxValue: Double = Double.MaxValue)

class MyRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[Instrument])
  }
}

object MonteCarloRisk {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Monte Carlo Risk")
    sparkConf.set("spark.serializer", classOf[KryoSerializer].getName)
    sparkConf.set("spark.kryo.registrator", classOf[MyRegistrator].getName)
    val sc = new SparkContext(sparkConf)

    // Parse arguments and read input data
    val instruments = readInstruments(args(0))
    val numTrials = args(1).toInt
    val parallelism = args(2).toInt
    val factorMeans = readMeans(args(3))
    val factorCovariances = readCovariances(args(4))
    val seed = if (args.length > 5) args(5).toLong else System.currentTimeMillis()

    // Send all instruments to every node
    val broadcastInstruments = sc.broadcast(instruments)

    // Generate different seeds so that our simulations don't all end up with the same results
    val seeds = (seed until seed + parallelism)
    val seedRdd = sc.parallelize(seeds, parallelism)

    // Main computation: run simulations and compute aggregate return for each
    val trialsRdd = seedRdd.flatMap(trialValues(_, numTrials / parallelism,
      broadcastInstruments.value, factorMeans, factorCovariances))

    // Cache the results so that we don't recompute for both of the summarizations below
    trialsRdd.cache()

    // Calculate VaR
    val varFivePercent = trialsRdd.takeOrdered(math.max(numTrials / 20, 1)).last
    println("VaR: " + varFivePercent)

    // Kernel density estimation
    val domain = Range.Double(20.0, 60.0, .2).toArray
    val densities = KernelDensity.estimate(trialsRdd, 0.25, domain)
    val pw = new PrintWriter("densities.csv")
    for (point <- domain.zip(densities)) {
      pw.println(point._1 + "," + point._2)
    }
    pw.close()
  }

  def trialValues(seed: Long, numTrials: Int, instruments: Seq[Instrument],
      factorMeans: Array[Double], factorCovariances: Array[Array[Double]]): Seq[Double] = {
    val rand = new MersenneTwister(seed)
    val multivariateNormal = new MultivariateNormalDistribution(rand, factorMeans,
      factorCovariances)

    val trialValues = new Array[Double](numTrials)
    for (i <- 0 until numTrials) {
      val trial = multivariateNormal.sample()
      trialValues(i) = trialValue(trial, instruments)
    }
    trialValues
  }

  /**
   * Calculate the full value of the portfolio under particular trial conditions.
   */
  def trialValue(trial: Array[Double], instruments: Seq[Instrument]): Double = {
    var totalValue = 0.0
    for (instrument <- instruments) {
      totalValue += instrumentTrialValue(instrument, trial)
    }
    totalValue
  }

  /**
   * Calculate the value of a particular instrument under particular trial conditions.
   */
  def instrumentTrialValue(instrument: Instrument, trial: Array[Double]): Double = {
    var instrumentTrialValue = 0.0
    var i = 0
    while (i < trial.length) {
      instrumentTrialValue += trial(i) * instrument.factorWeights(i)
      i += 1
    }
    Math.min(Math.max(instrumentTrialValue, instrument.minValue), instrument.maxValue)
  }

  def readInstruments(instrumentsFile: String): Array[Instrument] = {
    val src = Source.fromFile(instrumentsFile)
    // First and second elements are the min and max values for the instrument
    val instruments = src.getLines().map(_.split(",")).map(
      x => new Instrument(x.slice(2, x.length).map(_.toDouble), x(0).toDouble, x(1).toDouble))
    instruments.toArray
  }

  def readMeans(meansFile: String): Array[Double] = {
    val src = Source.fromFile(meansFile)
    val means = src.getLines().map(_.toDouble)
    means.toArray
  }

  def readCovariances(covsFile: String): Array[Array[Double]] = {
    val src = Source.fromFile(covsFile)
    val covs = src.getLines().map(_.split(",")).map(_.map(_.toDouble))
    covs.toArray
  }
}
