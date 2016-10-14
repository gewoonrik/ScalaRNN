package neuralnet.layers

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray

object SoftmaxBackProp extends BackProp {

  override def backProp(l: Layer, inputs: List[INDArray], outputs: List[INDArray], outputMasks: List[Boolean], gradientsNextLayer: List[INDArray], learningRate: Double): List[INDArray] = {
    // :'(
    val layer = l.asInstanceOf[SoftmaxLayer]
    //these only contain the masked gradients and inputs
    val gradientsM = gradientsNextLayer.zip(outputMasks).filter(_._2).map(_._1)
    val inputsM = inputs.zip(outputMasks).filter(_._2).map(_._1)


    val dVs = gradientsM.zip(inputsM).map(x => x._1.T ** x._2)
    val dBias = gradientsM.reduce(_+_)

    val dInputs: List[INDArray] = gradientsNextLayer.map(layer.V.T * _)

    val sumDVs = dVs.reduce[INDArray](_ + _)
    layer.V +=  preProcessGradients(sumDVs * -learningRate)
    layer.bias += preProcessGradients(dBias * -learningRate)

    dInputs
  }

}
