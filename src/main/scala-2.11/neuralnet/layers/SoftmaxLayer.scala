package neuralnet.layers

import neuralnet.LinAlgHelper
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._



class SoftmaxLayer(override val nrOfInputs : Int, override val nrOfOutputs : Int) extends Layer {


  override val backPropImpl = SoftmaxBackProp

  val V = initXavier(nrOfOutputs, nrOfInputs)

  val bias = initXavier(nrOfOutputs)


  override def forwardPass(x: INDArray) : INDArray = {
    softmax(V ** x + bias)
  }

  private def softmax(x : INDArray) : INDArray = {
    val eX = Transforms.exp(x)
    val s = eX.sumNumber().doubleValue()
    eX/s
  }


  override def reset: Unit = {}

}
