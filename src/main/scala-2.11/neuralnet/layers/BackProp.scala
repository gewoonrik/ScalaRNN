package neuralnet.layers

import breeze.linalg.Vector

trait BackProp  {
  def backProp(layer: Layer, inputs: List[Vector[Double]], outputs: List[Vector[Double]], outputMasks: List[Boolean], gradientsNextLayer: List[Vector[Double]], learningRate: Double): List[Vector[Double]]
}
