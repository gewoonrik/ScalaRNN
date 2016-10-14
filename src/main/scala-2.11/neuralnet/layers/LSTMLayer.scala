package neuralnet.layers

import neuralnet.ActivationFunction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

class LSTMLayer(override val nrOfInputs : Int, override val nrOfOutputs : Int, val activationFunction: ActivationFunction) extends Layer {
  override val backPropImpl: BackProp = ???

  val sigmoid = ActivationFunction.Sigmoid
  val tanh = ActivationFunction.TanH

  val F = initXavier(nrOfOutputs, nrOfInputs+nrOfOutputs)
  val fBias = initXavier(nrOfInputs+nrOfOutputs)

  val I = initXavier(nrOfOutputs, nrOfInputs+nrOfOutputs)
  val iBias = initXavier(nrOfOutputs)

  val C = initXavier(nrOfOutputs, nrOfInputs+nrOfOutputs)
  val cBias = initXavier(nrOfInputs+nrOfOutputs)

  val O = initXavier(nrOfOutputs, nrOfOutputs)
  val oBias = initXavier(nrOfOutputs)

  private var prevOutput : INDArray = Nd4j.zeros(nrOfOutputs,1)

  private var memory = Nd4j.zeros(nrOfOutputs, 1)


  private def forgetGateLayer(hiddenConcatInput : INDArray) : INDArray = {
    sigmoid(F ** hiddenConcatInput + fBias) * memory
  }

  private def inputGateLayer(hiddenConcatInput : INDArray)  : INDArray = {
    val i = sigmoid(I ** hiddenConcatInput + iBias) //this decides how much of the new input we are going to keep
    i * tanh(C ** hiddenConcatInput + cBias) // the actual new input
  }


  override def forwardPass(x: INDArray): INDArray = {
    val concat = Nd4j.concat(1, prevOutput, x)
    val forget = forgetGateLayer(concat)
    val input = inputGateLayer(concat)
    memory =  forget+input

    val oT = sigmoid(O ** concat + oBias)
    prevOutput = oT * tanh(memory)
    prevOutput
  }

  override def reset: Unit = {
    prevOutput = Nd4j.zeros(nrOfOutputs, 1)
    memory = Nd4j.zeros(nrOfOutputs, 1)
  }
}
