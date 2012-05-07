package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configurable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public interface VectorSparsifier extends Configurable {
  
  Vector sparsify(Vector input);
  
  Matrix rebalance(Matrix input);
  
}
