package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configured;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public class NoopVectorSparsifier extends Configured implements  VectorSparsifier {
  @Override
  public Vector sparsify(Vector input) {
    return input;
  }

  @Override
  public Matrix rebalance(Matrix input) {
    return input;
  }
}
