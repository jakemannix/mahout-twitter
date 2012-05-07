package org.apache.mahout.clustering.lda.cvb;


import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.VectorFunctions;
import org.junit.Test;

import static junit.framework.Assert.assertEquals;
import static junit.framework.Assert.assertTrue;

public class TestVectorSparsifier {

  @Test
  public void testBackgroundFrequencyVectorSparsifier() throws Exception {
    int numTopics = 10;
    int numFeatures = 100;
    Matrix original = ClusteringTestUtils.randomStructuredModel(numTopics, numFeatures);

    Vector backgroundCollectionFrequency = original.aggregateColumns(VectorFunctions.NORM_1);
    
    BackgroundFrequencyVectorSparsifier vectorSparsifier = 
        new BackgroundFrequencyVectorSparsifier(backgroundCollectionFrequency, 1);
    
    Matrix sparsified = new DenseMatrix(numTopics, numFeatures);
    
    for (int topic = 0; topic < numTopics; topic++) {
      sparsified.assignRow(topic, vectorSparsifier.sparsify(original.viewRow(topic)));
    }
    
    for (int feature = 0; feature < numFeatures; feature++) {
      Vector originalFeatureVector = original.viewColumn(feature);
      Vector sparsifiedFeatureVector = sparsified.viewColumn(feature);
      assertTrue("feature norms should shrink", 
                  originalFeatureVector.norm(1) > sparsifiedFeatureVector.norm(1));
      assertTrue("feature norms should be greater than 0",
                  sparsifiedFeatureVector.norm(1) > 0);
    }
    
    Matrix rebalanced = vectorSparsifier.rebalance(sparsified);

    for (int feature = 0; feature < numFeatures; feature++) {
      Vector originalFeatureVector = original.viewColumn(feature);
      Vector sparsifiedFeatureVector = rebalanced.viewColumn(feature);
      assertEquals("feature norms should be equal",
                   originalFeatureVector.norm(1),
                   sparsifiedFeatureVector.norm(1),
                   0.00001);
      int numNonZero = 0;
      for(Vector.Element e : sparsifiedFeatureVector) {
        if (e.get() != 0) {
          numNonZero++;
        }
      }
      assertTrue("at least half of the elements should be zero after sparsification",
                 numNonZero < sparsifiedFeatureVector.size() / 2);
      assertTrue("at least two of the elements should be > 0 after sparsification",
                 numNonZero > 0);
    }
    

  }
  
}
