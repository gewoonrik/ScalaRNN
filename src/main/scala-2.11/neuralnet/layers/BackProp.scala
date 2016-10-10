package neuralnet.layers

import breeze.linalg.Vector

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
  def backProp(layer: Layer, inputs: List[Vector[Double]], outputs: List[Vector[Double]], outputMasks: List[Boolean], gradientsNextLayer: List[Vector[Double]], learningRate: Double): List[Vector[Double]]
}
