package neuralnet
import breeze.linalg.{min, max, Matrix, Vector}
import breeze.numerics.sigmoid


trait ActivationFunction {
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