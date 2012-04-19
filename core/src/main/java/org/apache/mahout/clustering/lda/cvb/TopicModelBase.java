package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DistributedRowMatrixWriter;
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

  public void reset() {
    // NOOP
  }

  public void awaitTermination() {
    // NOOP
  }
  
  abstract public Matrix getTopicTermCounts();

  abstract public Vector expectedTermCounts(Vector original, Vector docTopics);

  public void persist(Path outputDir, boolean overwrite) throws IOException {
    FileSystem fs = outputDir.getFileSystem(getConf());
    if(overwrite) {
      fs.delete(outputDir, true); // CHECK second arg
    }
    DistributedRowMatrixWriter.write(outputDir, conf, getTopicTermCounts());
  }


  public static Pair<Matrix,Vector> randomMatrix(int numTopics, int numTerms, Random random) {
    Matrix topicTermCounts = new DenseMatrix(numTopics, numTerms);
    Vector topicSums = new DenseVector(numTopics);
    if(random != null) {
      for(int x = 0; x < numTopics; x++) {
        for(int term = 0; term < numTerms; term++) {
          topicTermCounts.viewRow(x).set(term, random.nextDouble());
        }
      }
    }
    for(int x = 0; x < numTopics; x++) {
      topicSums.set(x, random == null ? 1.0 : topicTermCounts.viewRow(x).norm(1));
    }
    return Pair.of(topicTermCounts, topicSums);
  }
}
