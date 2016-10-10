package neuralnet


/**
  *
  * @param outputMask is used to signal that we do or do not care about an output.
  *                   false means we do not care.
  *                   When false, no backpropogation is done for that output.
  *                   (This is used to support many to one RNN's)
  * @param labels     The actual labels
  */
case class Labels(outputMask : List[Boolean], labels: List[Int])

object Labels {
  def onlyOne(sequenceSize : Int, label : Int) = {
    val mask = (0 until sequenceSize).map(_=>false).toArray
    mask(sequenceSize-1) = true

    val labels = (0 until sequenceSize).map(_=>0).toArray
    labels(sequenceSize-1) = label

    Labels(mask.toList, labels.toList)
  }

  def full(labels : List[Int]) = {
    val mask = labels.map(_=>true)
    Labels(mask, labels)
  }
}
