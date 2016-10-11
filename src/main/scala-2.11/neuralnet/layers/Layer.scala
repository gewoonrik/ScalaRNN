package neuralnet.layers

import breeze.linalg._
import breeze.stats.distributions.Gaussian

trait Layer {

  val backPropImpl : BackProp

  def nrOfInputs: Int

  def nrOfOutputs: Int

  def initXavier(rows: Int, columns: Int): DenseMatrix[Double] = {
    val sigma = Math.sqrt(2 / (nrOfInputs.toDouble + nrOfOutputs))

    val normal = Gaussian(0, sigma)
    val samples = normal.sample(rows*columns)

    new DenseMatrix[Double](rows, columns, samples.toArray)
  }

  def initXavier(rows: Int): DenseVector[Double] = {
    val sigma = Math.sqrt(2 / (nrOfInputs.toDouble + nrOfOutputs))

    val normal = Gaussian(0, sigma)
    val samples = normal.sample(rows)

    new DenseVector[Double](samples.toArray)
  }

  def forwardPass(x: Vector[Double]): Vector[Double]


  def reset : Unit
}
