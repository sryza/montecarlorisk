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
import java.io.{PrintWriter, File}

case class Instrument(factorWeights: Array[Double])

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

    val instruments = readInstruments(args(0))
    val numTrials = args(1).toInt
    val parallelism = args(2).toInt
    val factorMeans = readMeans(args(3))
    val factorCovariances = readCovariances(args(4))
    val seed = if (args.length > 5) args(5).toLong else System.currentTimeMillis()

    val broadcastInstruments = sc.broadcast(instruments)
    val seeds = (seed until seed + parallelism)

    val seedRdd = sc.parallelize(seeds, parallelism)
    val trialsRdd = seedRdd.flatMap(trialReturns(_, numTrials / parallelism,
      broadcastInstruments.value, factorMeans, factorCovariances))

//    val sorted = trialsRdd.map(t => (t, None)).sortByKey()
    //val trials = trialsRdd.collect()
//    val pw = new PrintWriter(new File("returns.json"))
//    pw.println("[" + trials.mkString(",") + "]")
    //pw.close()
    val varFivePercent = trialsRdd.takeOrdered(numTrials / 20).last
    println("VaR: " + varFivePercent)
  }

  def trialReturns(seed: Long, numTrials: Int, instruments: Seq[Instrument],
      factorMeans: Array[Double], factorCovariances: Array[Array[Double]]): Seq[Double] = {
    val rand = new MersenneTwister(seed)
    val multivariateNormal = new MultivariateNormalDistribution(rand, factorMeans,
      factorCovariances)

    val trialReturns = new Array[Double](numTrials)
    for (i <- 0 until numTrials) {
      val trial = multivariateNormal.sample()
      trialReturns(i) = trialReturn(trial, instruments)
    }
    trialReturns
  }

  def trialReturn(trial: Array[Double], instruments: Seq[Instrument]): Double = {
    var totalReturn = 0.0
    for (instrument <- instruments) {
      totalReturn += instrumentTrialReturn(instrument, trial)
    }
    totalReturn
  }

  def instrumentTrialReturn(instrument: Instrument, trial: Array[Double]): Double = {
    var instrumentTrialReturn = 0.0
    var i = 0
    while (i < trial.length) {
      instrumentTrialReturn += trial(i) * instrument.factorWeights(i)
      i += 1
    }
    instrumentTrialReturn
  }

  def readInstruments(instrumentsFile: String): Array[Instrument] = {
    val src = Source.fromFile(instrumentsFile)
    val instruments = src.getLines().map(_.split(",")).map(x => new Instrument(x.map(_.toDouble)))
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
