package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;

import org.apache.hadoop.conf.Configurable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public interface VectorSparsifier extends Configurable {

  /**
   * Transform a vector by pruning out non-informative elements.
   * @param input a vector (most likely dense) to be sparsified
   * @return a vector (most likely sparse) which has been sparsified
   */
  Vector sparsify(Vector input);

  /**
   * Take a matrix which has been sparsified, and make sure that the columns sum up to what
   * they summed up to before.
   * @param input
   * @return
   */
  Matrix rebalance(Matrix input);

  /**
   * load any require auxiliary data
   * @throws IOException
   */
  void initialize() throws IOException;
  
}
