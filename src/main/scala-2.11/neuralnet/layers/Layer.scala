package neuralnet.layers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._


trait Layer {

  val backPropImpl : BackProp

  def nrOfInputs: Int

  def nrOfOutputs: Int

  def initXavier(rows: Int, columns: Int): INDArray = {
    val sigma = Math.sqrt(2 / (nrOfInputs.toDouble + nrOfOutputs))
    Nd4j.randn(rows, columns) * sigma
  }

  def initXavier(rows: Int): INDArray = {
    val sigma = Math.sqrt(2 / (nrOfInputs.toDouble + nrOfOutputs))


    Nd4j.randn(rows, 1) * sigma
  }

  def forwardPass(x: INDArray): INDArray


  def reset : Unit
}
