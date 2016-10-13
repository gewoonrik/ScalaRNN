package neuralnet.layers

import breeze.linalg
import breeze.linalg.Tensor
import org.nd4j.linalg.api.ndarray.INDArray

object LSTMBackProp extends BackProp{
  /**
    * Does backpropagation
    * @param layer the layer that is being updated.
    * @param inputs all the inputs to that layer of all timesteps.
    * @param outputs all the outputs of that layer of all timesteps.
    * @param outputMasks the outputmasks of the labels.
    * @param gradientsNextLayer the gradients of the next layer.
    * @param learningRate the learningrate.
    * @return
    */
  override def backProp(layer: Layer, inputs: List[INDArray], outputs: List[INDArray], outputMasks: List[Boolean], gradientsNextLayer: List[INDArray], learningRate: Double): List[INDArray] = ???

}
