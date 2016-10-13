package neuralnet
import breeze.linalg.Vector
import breeze.numerics.{sech, sigmoid, tanh}


trait ActivationFunction {
  def apply(value : Vector[Double]) : Vector[Double] = call(value)
  def call(value : Vector[Double]): Vector[Double]
  def derivative(value : Vector[Double]) : Vector[Double]
}

object ActivationFunction {
  object Sigmoid extends ActivationFunction {
    override def call(value: Vector[Double]): Vector[Double] = {
      //1/(1+Math.exp(-value))
      sigmoid(value)
    }
    override def derivative(value: Vector[Double]): Vector[Double] = {
      val sig = call(value)
      sig :* (sig * -1.0 + 1.0)
    }
  }

  object TanH extends ActivationFunction {
    override def call(value: Vector[Double]): Vector[Double] = tanh(value)

    override def derivative(value: Vector[Double]): Vector[Double] = {
      val temp = sech(value)
      temp :*= temp
    }
  }

  object ReLu extends ActivationFunction {
    override def call(value: Vector[Double]): Vector[Double] = {
      value.map(x => Math.max(x,0.0))
    }
    override def derivative(value: Vector[Double]): Vector[Double] = {
      value
        .map(x => Math.max(x,0.0))
        .map(x => Math.min(x, 1.0))
    }
  }
}