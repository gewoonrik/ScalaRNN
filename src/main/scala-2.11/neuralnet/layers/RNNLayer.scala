package neuralnet.layers

import neuralnet.ActivationFunction
import breeze.linalg._


class RNNLayer(override val nrOfInputs : Int, override val nrOfOutputs : Int, val activationFunction: ActivationFunction) extends Layer {


  override val backPropImpl = RNNBackProp

  //input weights
  val U = initXavier(nrOfOutputs, nrOfInputs)

  //recurrent weights
  val W = initXavier(nrOfOutputs, nrOfOutputs)

  val bias = initXavier(nrOfOutputs)

  var hiddenState = Vector.zeros[Double](nrOfOutputs)


  //this is one step in the RNN
  def forwardPass(input: Vector[Double]): Vector[Double] = {
    hiddenState = activationFunction.call(U * input + W * hiddenState + bias)
    hiddenState
  }

  /**
    * resets the state of this RNN layer
    * @return
    */
  def reset: Unit = {
    hiddenState = Vector.zeros[Double](nrOfOutputs)
  }

}
