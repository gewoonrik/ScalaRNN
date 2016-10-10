package neuralnet.layers

import neuralnet.LinAlgHelper
import breeze.linalg.Vector

class SoftmaxBackProp extends BackProp {

  override def backProp(l: Layer, inputs: List[Vector[Double]], outputs: List[Vector[Double]], outputMasks: List[Boolean], gradientsNextLayer: List[Vector[Double]], learningRate: Double): List[Vector[Double]] = {
    val layer = l.asInstanceOf[SoftmaxLayer]
    val dVs = gradientsNextLayer.zip(inputs).map((LinAlgHelper.outerProduct _).tupled)
    val dBias = gradientsNextLayer.reduce(_+_)

    val dInputs : List[Vector[Double]] = dVs.zip(gradientsNextLayer).map(x=> x._1.t * x._2)

    layer.V +=  -learningRate * dVs.reduce(_+_)
    layer.bias += -learningRate * dBias

    dInputs
  }
}
