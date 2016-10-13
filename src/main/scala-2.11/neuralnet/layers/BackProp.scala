package neuralnet.layers

import breeze.linalg.support._
import breeze.linalg._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

trait BackProp  {
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
  def backProp(layer: Layer, inputs: List[INDArray], outputs: List[INDArray], outputMasks: List[Boolean], gradientsNextLayer: List[INDArray], learningRate: Double): List[INDArray]


  def preProcessGradients(vector: INDArray) : INDArray = {
    Transforms.max(vector, -2)
  }
}
