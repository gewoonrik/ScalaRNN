package neuralnet.layers

import neuralnet.LinAlgHelper
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

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
  override def backProp(l : Layer, inputs: List[INDArray], outputs: List[INDArray], outputMasks : List[Boolean], gradientsNextLayer: List[INDArray], learningRate: Double) : List[INDArray] = {
    //yes, I cry in my sleep because of this.
    //I need to find a typesafe way to do this.
    val layer = l.asInstanceOf[RNNLayer]
    val W = layer.W
    val U = layer.U
    val bias = layer.bias

    val nrOfInputs = layer.nrOfInputs
    val nrOfOutputs = layer.nrOfOutputs
    val activationFunction = layer.activationFunction


    val dW = Nd4j.zeros(W.rows, W.columns())
    val dU = Nd4j.zeros(U.rows, U.columns())
    val dBias = Nd4j.zeros(bias.length, 1)



    //add the first empty hiddenstate to all outputs
    val hiddenStates = Nd4j.zeros(layer.nrOfOutputs, 1) :: outputs

    //contains all gradients with respect to the input, per timestep
    val inputGradientsSummed : Array[INDArray] = new Array[INDArray](outputs.length).map(_ => Nd4j.zeros(layer.nrOfInputs, 1))

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
    layer.W += preProcessGradients(dW * -learningRate)
    layer.U += preProcessGradients(dU * -learningRate)
    layer.bias += preProcessGradients(dBias * -learningRate)

    //timesteps were processed in reverse order, so reverse again
    inputGradientsSummed.toList.reverse
  }

  /**
    * This method updates arr1
    * @param arr1
    * @param arr2
    */
  private def sumPerTimestep(arr1 : Array[INDArray], arr2 : Array[INDArray]) = {
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
  private def backPropOneOutput(layer : RNNLayer,dW : INDArray, dU : INDArray, dBias : INDArray,
                                outputs: List[INDArray], inputs: List[INDArray],
                                gradientNextLayer: INDArray): Array[INDArray] = {
    val hiddenStates = outputs.reverse
    val rInputs = inputs.reverse
    val curHiddenStates = hiddenStates.view.dropRight(1)
    val prevHiddenStates = hiddenStates.view.drop(1)

    //the first recurrent delta is that of the outputlayer
    var deltaTime = gradientNextLayer

    val inputGradients = new Array[INDArray](curHiddenStates.length)

    for ((((output, prevHidden), input), i) <- curHiddenStates.zip(prevHiddenStates).zip(rInputs).zipWithIndex) {
      /** Hidden **/
      //dHidden = gradientNextLayer + the delta of "next" step backpropagated
      //        = gradientNextLayer + deltaTime
      val dActivation = layer.activationFunction.derivative(output) * deltaTime

      //dActivation * dActivation/biasHidden
      //dActivation/biasHidden = 1
      dBias += dActivation

      //dActivation * dActivation/dW
      //dActivation/dW = prevHiddenState
      dW += dActivation.T ** prevHidden


      /** Input **/
      //dActivation/dCurHiddenState = prevInput
      dU += dActivation.T ** input


      //dActivation/dCurHiddenState = W
      deltaTime = layer.W ** dActivation
      inputGradients(i) = layer.U.T ** dActivation
    }
    inputGradients
  }
}
