package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DistributedRowMatrixWriter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public abstract class TopicModelBase implements Configurable {

  private Configuration conf;
  protected final int numTopics;
  
  protected TopicModelBase(int numTopics) {
    this.numTopics = numTopics;
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
  
  abstract public void update(DocTrainingState state);

  abstract public Vector infer(DocTrainingState state, double minRelPerplexityDiff, int maxIters);

  public int getNumNonZeroes() {
    return 0;
  }
  
  public void trainDocTopicModel(DocTrainingState state) {
    for(int iter = 0; iter < state.getMaxIters(); iter++) {
      trainDocTopicModelSingleIteration(state);
    }
  }

  abstract protected void trainDocTopicModelSingleIteration(DocTrainingState state);

  abstract public double perplexity(Vector document, Vector docTopics);

  public void reset() {
    // NOOP
  }

  public void awaitTermination() {
    // NOOP
  }
  
  abstract public Iterable<MatrixSlice> getTopicVectors();

  abstract public Vector expectedTermCounts(Vector original, Vector docTopics);

  public void persist(Path outputDir, boolean overwrite) throws IOException {
    FileSystem fs = outputDir.getFileSystem(getConf());
    if(overwrite) {
      fs.delete(outputDir, true); // CHECK second arg
    }
    DistributedRowMatrixWriter.write(outputDir, conf, getTopicVectors());
  }

  public static Pair<Matrix, Vector> loadModel(Configuration conf, Path... modelPaths)
      throws IOException {
    int numTopics = -1;
    int numTerms = -1;
    List<Pair<Integer, Vector>> rows = Lists.newArrayList();
    for(Path modelPath : modelPaths) {
      for(Pair<IntWritable, VectorWritable> row :
          new SequenceFileIterable<IntWritable, VectorWritable>(modelPath, true, conf)) {
        rows.add(Pair.of(row.getFirst().get(), row.getSecond().get()));
        numTopics = Math.max(numTopics, row.getFirst().get());
        if(numTerms < 0) {
          numTerms = row.getSecond().get().size();
        }
      }
    }
    if(rows.isEmpty()) {
      throw new IOException(Arrays.toString(modelPaths) + " have no vectors in it");
    }
    numTopics++;
    Matrix model = new DenseMatrix(numTopics, numTerms);
    Vector topicSums = new DenseVector(numTopics);
    for(Pair<Integer, Vector> pair : rows) {
      model.viewRow(pair.getFirst()).assign(pair.getSecond());
      topicSums.set(pair.getFirst(), pair.getSecond().norm(1));
    }
    return Pair.of(model, topicSums);
  }

  public static Pair<Matrix,Vector> randomMatrix(int numTopics, int numTerms, Random random) {
    return randomMatrix(numTopics, numTerms, 1.0, random);
  }

  public static Pair<Matrix,Vector> randomMatrix(int numTopics, int numTerms, double weight,
    Random random) {
    Matrix topicTermCounts = new DenseMatrix(numTopics, numTerms);
    Vector topicSums = new DenseVector(numTopics);
    if(random != null) {
      for(int x = 0; x < numTopics; x++) {
        for(int term = 0; term < numTerms; term++) {
          topicTermCounts.viewRow(x).set(term, random.nextDouble() * weight);
        }
      }
    }
    for(int x = 0; x < numTopics; x++) {
      topicSums.set(x, random == null ? 1.0 : topicTermCounts.viewRow(x).norm(1));
    }
    return Pair.of(topicTermCounts, topicSums);
  }
}
