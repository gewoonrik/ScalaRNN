import java.io.File

import breeze.linalg._
import neuralnet.layers.{RNNLayer, SoftmaxLayer}
import neuralnet.{Labels, ActivationFunction, Network}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


object Main {
  def main(args: Array[String]) : Unit = {
    val vocab = loadModelVec(new File("/Users/rik/Downloads/aclImdb/model.vec"))

    val totalSize = 2*2000
    val epochs = 10
    val trainingPercentage = 0.95
    val testPercentage = 1-trainingPercentage
    val vocabVectorSize = vocab.head._2.length

    def toInputArray(x : List[Array[Double]]) : List[INDArray] = {
      x.map {
        w =>
          Nd4j.create(w)
      }
    }

    println("vocabulary size:"+ vocab.size)
    val neg = loadData(new File("/Users/rik/Downloads/aclImdb/test/neg"), totalSize/2).map((_,false))
    val pos = loadData(new File("/Users/rik/Downloads/aclImdb/test/pos"), totalSize/2).map((_, true))
    val alternating = new AlternatingIterator[(String,Boolean)](neg.iterator, pos.iterator).toList

    println("loaded files")
    val sequences = alternating
      .map(x => (x._1.split(" ").toList, x._2))
      .map(x =>
        (x._1
          .filter(w=>vocab.contains(w))
          .map(w=> vocab.get(w).get),
        x._2)
      )

    val network = new Network()
      .andThen(new RNNLayer(vocabVectorSize,200, ActivationFunction.ReLu))
      .andThen(new RNNLayer(200,200, ActivationFunction.ReLu))
      .andThen(new SoftmaxLayer(200,2))

    //take data from outside the trainingset
    val testSet = sequences.slice((trainingPercentage*totalSize).toInt , totalSize - 1).map(x => (toInputArray(x._1), x._2))
    val labels = testSet.map(x => if(x._2) 1 else 0)

    println("start learning")

    val trainingSet = sequences.take((trainingPercentage*totalSize).toInt)
    for(epoch <- 0 until epochs) {
      for (((sequence, pos), index) <- trainingSet.view.zipWithIndex) {

        if (index % 100 == 0) {
          val loss = network.calculateLoss(testSet.map(_._1), labels)
          println("iteration " + index + ": loss: " + loss)
        }
        val input = toInputArray(sequence)
        val label = if (pos) 1 else 0
        network.SGD(input, Labels.onlyOne(sequence.length, label))

      }
    }
  }




  def loadVocab(file : File): Map[String, Int] = {
    val words =  "BEGIN" :: "END" :: scala.io.Source.fromFile(file)
      .getLines.toList

    words.zipWithIndex
      .map(x => x._1 -> x._2).toMap
  }

  def loadModelVec(file : File) = {
    scala.io.Source.fromFile(file).getLines()
      .map(_.split(" "))
      .map(x => x.head -> x.tail.map(_.toDouble)).toMap
  }


  def loadData(dir : File, count: Int) : List[String] =  {
    dir.listFiles().filter(_.isFile).take(count)
      .map(scala.io.Source.fromFile)
      .map("BEGIN "+_.getLines().mkString.toLowerCase+ " END")
      .map(_.replace("<br />", "").replace("  ", " ").replaceAll("""\p{Punct}""", "")) //remove punctuation
      .toList
  }

  class AlternatingIterator[T](it1 : Iterator[T], it2: Iterator[T]) extends Iterator[T] {
    override def hasNext: Boolean = it1.hasNext || it2.hasNext

    private var one = true
    override def next(): T = {
      one = !one
      if(!it1.hasNext)
        it2.next
      else if(!it2.hasNext)
        it1.next
      else if(one)
        it1.next
      else
        it2.next
    }
  }
}
