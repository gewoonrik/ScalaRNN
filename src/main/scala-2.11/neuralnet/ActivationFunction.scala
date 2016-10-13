package neuralnet

import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative


trait ActivationFunction {
  def apply(value : INDArray) : INDArray = call(value)
  def call(value : INDArray): INDArray
  def derivative(value : INDArray) : INDArray
}

object ActivationFunction {
  object Sigmoid extends ActivationFunction {
    override def call(value: INDArray): INDArray = {
      //1/(1+Math.exp(-value))
      Transforms.sigmoid(value)
    }
    override def derivative(value: INDArray): INDArray = {
      val sig = call(value)
      sig * (sig * -1.0 + 1.0)
    }
  }

  object TanH extends ActivationFunction {
    override def call(value: INDArray): INDArray = Transforms.tanh(value)

    override def derivative(value: INDArray): INDArray = {
      val ret = value.dup()
      Nd4j.getExecutioner.exec(new TanhDerivative(value, ret))
      ret
    }
  }

  object ReLu extends ActivationFunction {
    override def call(value: INDArray): INDArray = {
      Transforms.max(value, 0.0)
    }

    override def derivative(value: INDArray): INDArray = {
      val ret = value.dup()
      Nd4j.getExecutioner.exec(new Min(value, ret,1))
      Transforms.max(ret, 0.0)
    }
  }
}