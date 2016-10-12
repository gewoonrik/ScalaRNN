package neuralnet.layers

import neuralnet.LinAlgHelper
import breeze.linalg._

object RNNBackProp extends BackProp{
  /***
    *
    * @param l
    * @param inputs inputs, in order of input
    * @param outputs outputs, in order of input
    * @param outputMasks
    * @param gradientsNextLayer
    * @return
    */
  override def backProp(l : Layer, inputs: List[Vector[Double]], outputs: List[Vector[Double]], outputMasks : List[Boolean], gradientsNextLayer: List[Vector[Double]], learningRate: Double) : List[Vector[Double]] = {
    //yes, I cry in my sleep because of this.
    //I need to find a typesafe way to do this.
    val layer = l.asInstanceOf[RNNLayer]
    val W = layer.W
    val U = layer.U
    val bias = layer.bias

    val nrOfInputs = layer.nrOfInputs
    val nrOfOutputs = layer.nrOfOutputs
    val activationFunction = layer.activationFunction


    val dW = DenseMatrix.zeros[Double](W.rows, W.cols)
    val dU = DenseMatrix.zeros[Double](U.rows, U.cols)
    val dBias = DenseVector.zeros[Double](bias.length)



    //add the first empty hiddenstate to all outputs
    val hiddenStates = DenseVector.zeros[Double](layer.nrOfOutputs) :: outputs

    //contains all gradients with respect to the input, per timestep
    val inputGradientsSummed : Array[Vector[Double]] = new Array[Vector[Double]](outputs.length).map(_ => DenseVector.zeros[Double](layer.nrOfInputs))

    for(((gradient, outputMask),i) <-
        gradientsNextLayer
          .zip(outputMasks)
          .zipWithIndex) {
      if(outputMask) {
        val inpGradients = backPropOneOutput(layer, dW, dU, dBias, hiddenStates.take(i+2), inputs.take(i+1), gradient)
        sumPerTimestep(inputGradientsSummed, inpGradients)
      }
    }
    //update parameters
    layer.W += preProcessGradients(-learningRate * dW)
    layer.U += preProcessGradients(-learningRate * dU)
    layer.bias += preProcessGradients(-learningRate * dBias)

    inputGradientsSummed.toList
  }

  /**
    * This method updates arr1
    * @param arr1
    * @param arr2
    */
  private def sumPerTimestep(arr1 : Array[Vector[Double]], arr2 : Array[Vector[Double]]) = {
    //arr 2 has less steps, so step 0 in arr2 != step 0 in arr1
    val lengthDiff = arr1.length - arr2.length
    for((vec, index) <- arr2.zipWithIndex) {
      arr1(index+lengthDiff) += vec
    }
  }

  /**
    * Does the backpropagation for one output
    * This method has a lot of side effects :)
    * @param layer
    * @param dW
    * @param dU
    * @param dBias
    * @param outputs
    * @param inputs
    * @param gradientNextLayer
    * @return the gradients per timestep with respect to the input
    */
  private def backPropOneOutput(layer : RNNLayer,dW : Matrix[Double], dU : Matrix[Double], dBias : Vector[Double],
                                outputs: List[Vector[Double]], inputs: List[Vector[Double]],
                                gradientNextLayer: Vector[Double]): Array[Vector[Double]] = {
    val hiddenStates = outputs.reverse
    val rInputs = inputs.reverse
    val curHiddenStates = hiddenStates.view.dropRight(1)
    val prevHiddenStates = hiddenStates.view.drop(1)

    //the first recurrent delta is that of the outputlayer
    var deltaTime : Vector[Double] = gradientNextLayer

    val inputGradients = new Array[Vector[Double]](curHiddenStates.length)

    for ((((output, prevHidden), input), i) <- curHiddenStates.zip(prevHiddenStates).zip(rInputs).zipWithIndex) {
      /** Hidden **/
      //dHidden = gradientNextLayer + the delta of "next" step backpropagated
      //        = gradientNextLayer + deltaTime
      val dActivation = layer.activationFunction.derivative(output) :* deltaTime

      //dActivation * dActivation/biasHidden
      //dActivation/biasHidden = 1
      dBias += dActivation

      //dActivation * dActivation/dW
      //dActivation/dW = prevHiddenState
      dW += LinAlgHelper.outerProduct(dActivation, prevHidden)


      /** Input **/
      //dActivation/dCurHiddenState = prevInput
      dU += LinAlgHelper.outerProduct(dActivation, input)


      //dActivation/dCurHiddenState = W
      deltaTime = layer.W * dActivation
      //todo: I think this is not in correct order
      inputGradients(i) = layer.U.toDenseMatrix.t * dActivation
    }
    inputGradients
  }
}
