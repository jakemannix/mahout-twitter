package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public abstract class TopicModelBase implements Configurable {

  private Configuration conf;
  protected final int numTopics;
  private final Vector uniform;
  
  protected TopicModelBase(int numTopics) {
    this.numTopics = numTopics;
    uniform = new DenseVector(numTopics).assign(1.0 / numTopics);
  }

  @Override
  public void setConf(Configuration configuration) {
    this.conf = configuration;
  }

  @Override
  public Configuration getConf() {
    return conf;
  }
  
  public final int getNumTopics() {
    return numTopics;
  }

  public final Vector getUniform() {
    return uniform;
  }
  
  abstract public void update(DocTrainingState state);

  abstract public Vector infer(DocTrainingState state, double minRelPerplexityDiff, int maxIters);

  abstract public void trainDocTopicModel(DocTrainingState state);

  abstract public double perplexity(Vector document, Vector docTopics);

}
