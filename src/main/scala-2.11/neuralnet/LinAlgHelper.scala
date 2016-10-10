package neuralnet

import breeze.linalg._


object LinAlgHelper {
  /**
    * Too bad Breeze does not support this directly :(
    * @param v1
    * @param v2
    * @return
    */
  def outerProduct(v1 : Vector[Double], v2 : Vector[Double]) : DenseMatrix[Double] = {

    val v1d = v1 match {
      case x : SparseVector[Double] => x.toDenseVector
      case x : DenseVector[Double] => x
    }

    val v2d = v2 match {
      case x : SparseVector[Double] => x.toDenseVector
      case x : DenseVector[Double] => x
    }

    v1d.asDenseMatrix.t * v2d.asDenseMatrix
  }
}
