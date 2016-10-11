package neuralnet.layers

import neuralnet.LinAlgHelper
import breeze.linalg.{Matrix, Vector}

object SoftmaxBackProp extends BackProp {

  override def backProp(l: Layer, inputs: List[Vector[Double]], outputs: List[Vector[Double]], outputMasks: List[Boolean], gradientsNextLayer: List[Vector[Double]], learningRate: Double): List[Vector[Double]] = {
    // :'(
    val layer = l.asInstanceOf[SoftmaxLayer]
    //these only contain the masked gradients and inputs
    val gradientsM = gradientsNextLayer.zip(outputMasks).filter(_._2).map(_._1)
    val inputsM = inputs.zip(outputMasks).filter(_._2).map(_._1)


    val dVs = gradientsM.zip(inputsM).map(LinAlgHelper.outerProduct _ tupled)
    val dBias = gradientsM.reduce(_+_)

    val dInputs = gradientsNextLayer.map(layer.V.t.toDenseMatrix * _)

    layer.V +=  -learningRate * dVs.reduce(_+_)
    layer.bias += -learningRate * dBias

    dInputs
  }
}
