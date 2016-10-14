package neuralnet.layers

import neuralnet.ActivationFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._


class RNNLayer(override val nrOfInputs : Int, override val nrOfOutputs : Int, val activationFunction: ActivationFunction) extends Layer {


  override val backPropImpl = RNNBackProp

  //input weights
  val U = initXavier(nrOfOutputs, nrOfInputs)

  //recurrent weights
  val W = initXavier(nrOfOutputs, nrOfOutputs)

  val bias = initXavier(nrOfOutputs)

  var hiddenState = Nd4j.zeros(nrOfOutputs, 1)


  //this is one step in the RNN
  def forwardPass(input: INDArray): INDArray = {
    hiddenState = activationFunction.call(U ** input + W ** hiddenState + bias)
    hiddenState
  }

  /**
    * resets the state of this RNN layer
    * @return
    */
  def reset: Unit = {
    hiddenState = Nd4j.zeros(nrOfOutputs)
  }

}
