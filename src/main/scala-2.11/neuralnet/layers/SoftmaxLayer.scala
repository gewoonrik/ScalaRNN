package neuralnet.layers

import neuralnet.LinAlgHelper
import breeze.linalg._
import breeze.numerics.exp


class SoftmaxLayer(override val nrOfInputs : Int, override val nrOfOutputs : Int) extends Layer {


  override val backPropImpl = SoftmaxBackProp

  val V = initXavier(nrOfOutputs, nrOfInputs)

  val bias = initXavier(nrOfOutputs)


  override def forwardPass(x: Vector[Double]) : Vector[Double] = {
    softmax(V * x + bias)
  }

  private def softmax(x : Vector[Double]) : Vector[Double] = {
    val eX = exp(x)
    val s = sum(eX)
    eX/s
  }


  override def reset: Unit = {}

}
