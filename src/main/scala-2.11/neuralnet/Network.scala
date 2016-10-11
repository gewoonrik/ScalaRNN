package neuralnet

import neuralnet.layers.Layer
import breeze.linalg.{argmax, SparseVector, Vector}


case class Network(learningRate : Double = 0.02) {

  type InputVector = Vector[Double]
  type Sequence = List[InputVector]

  private var layers : List[Layer] = List()

  /**
    * Adds a layer to the network
    * @param layer
    * @return returns this to support chaining
    */
  def andThen(layer : Layer) = {
    layers = (layer :: layers.reverse).reverse
    this
  }

  /**
    * Runs one step of the RNN
    * @param input
    * @return a List of the outputs of all layers in reverse. Last output first
    */
  def run(input : Vector[Double]) : List[Vector[Double]] =  {
    layers.foldLeft(List(input)){
      case (inpList, layer) =>
        layer.forwardPass(inpList.head) :: inpList
    }
  }

  /**
    * Runs a RNN on the whole sequence
    * @param input
    * @return The last output of the RNN sequence
    */
  def run(input : Sequence) : List[Vector[Double]] =  {
    input.map(run(_).head)
  }

  def predict(input : Sequence) : Int = {
    val output = input.map(run).last.last //this map is side effecting
    argmax(output)
  }

  /**
    * stochastic gradient descent
    * @param input
    * @param labels
    */
  def SGD(input : Sequence, labels: Labels)  = {
    val layersReverse = layers.reverse
    val outputs = input.map(run)
    val outputsPerLayer = outputs.flatMap(_.zipWithIndex).groupBy(x => x._2).toList.sortBy(_._1).map(x => x._2.map(y => y._1))


    var outputGradients =
      outputs
        .map(_.head)
    //softmax derative
    outputGradients.zip(labels.labels).foreach(x => x._1(x._2) -= 1)
    for(((layer, output),input) <- layersReverse.zip(outputsPerLayer.dropRight(1)).zip(outputsPerLayer.drop(1))) {
      outputGradients = layer.backPropImpl.backProp(layer, input, output, labels.outputMask, outputGradients, learningRate)
    }

  }

  def reset = {
    layers.foreach(_.reset)
  }

  /**
    * calculates the cross entropy loss
    * @param input
    * @param labels the ground truth of the final output of each sequence
    * @return
    */
  def calculateLoss(input : List[Sequence], labels : List[Int]) = {
    val N = labels.length.toDouble
    val outputs = input.map(x => {val out = run(x); reset; out}).map(_.last)
    -1/N * outputs.zip(labels)
      .map(x => x._1(x._2))
      .map(Math.log)
      .sum
  }
}
