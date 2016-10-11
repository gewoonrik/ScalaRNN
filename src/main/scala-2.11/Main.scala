import java.io.File

import breeze.linalg._
import neuralnet.layers.{RNNLayer, SoftmaxLayer}
import neuralnet.{Labels, ActivationFunction, Network}


object Main {
  def main(args: Array[String]) : Unit = {
    val vocab = loadVocab(new File("/Users/rik/Downloads/aclImdb/imdb.vocab"))

    val totalSize = 2*500
    val trainingPercentage = 0.9
    val testPercentage = 1-trainingPercentage

    def toInputArray(x : List[Int]) : List[Vector[Double]] = {
      x.map {
        w =>
          new SparseVector[Double](Array(w), Array(1.0), vocab.size)
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
      .andThen(new RNNLayer(vocab.size,100, ActivationFunction.ReLu))
      .andThen(new RNNLayer(100,50, ActivationFunction.ReLu))
      .andThen(new SoftmaxLayer(50,2))

    //take data from outside the trainingset
    val testSet = sequences.slice((trainingPercentage*totalSize).toInt , totalSize - 1).map(x => (toInputArray(x._1), x._2))
    val labels = testSet.map(x => if(x._2) 1 else 0)

    println("start learning")

    val trainingSet = sequences.take((trainingPercentage*totalSize).toInt)
    for(((sequence,pos), index) <- trainingSet.view.zipWithIndex) {

      if(index %10 == 0) {
        val loss = network.calculateLoss(testSet.map(_._1), labels)
        println("iteration "+index +": loss: "+loss)
      }

      val input = toInputArray(sequence)
      val label = if(pos) 1 else 0
      network.SGD(input, Labels.onlyOne(sequence.length,label))

    }
  }




  def loadVocab(file : File): Map[String, Int] = {
    val words =  "BEGIN" :: "END" :: scala.io.Source.fromFile(file)
      .getLines.toList

    words.zipWithIndex
      .map(x => x._1 -> x._2).toMap
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
