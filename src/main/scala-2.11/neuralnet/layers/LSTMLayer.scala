package neuralnet.layers

import breeze.linalg._
import neuralnet.ActivationFunction

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

  private var prevOutput : DenseVector[Double] = DenseVector.zeros[Double](nrOfOutputs)

  private var memory : Vector[Double] = DenseVector.zeros[Double](nrOfOutputs)


  private def forgetGateLayer(hiddenConcatInput : Vector[Double]) : Vector[Double] = {
    sigmoid(F * hiddenConcatInput + fBias) :* memory
  }

  private def inputGateLayer(hiddenConcatInput : Vector[Double])  : Vector[Double] = {
    val i = sigmoid(I * hiddenConcatInput + iBias) //this decides how much of the new input we are going to keep
    i :* tanh(C * hiddenConcatInput + cBias) // the actual new input
  }


  override def forwardPass(x: Vector[Double]): Vector[Double] = {
    val concat = DenseVector.vertcat[Double](prevOutput, x.toDenseVector)
    val forget = forgetGateLayer(concat)
    val input = inputGateLayer(concat)
    memory =  forget+input

    val oT = sigmoid(O * concat + oBias)
    prevOutput = (oT :* tanh(memory)).toDenseVector
    prevOutput
  }

  override def reset: Unit = {
    prevOutput = DenseVector.zeros(nrOfOutputs)
    memory = DenseVector.zeros(nrOfOutputs)
  }
}
