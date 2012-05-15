package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;
import java.util.Iterator;

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

public class BackgroundFrequencyVectorSparsifier extends Configured implements VectorSparsifier {
  private float minCfRatio = Float.NaN;
  private double[] collectionFrequencies = null;
  private double corpusWeight = -1;

  public BackgroundFrequencyVectorSparsifier() {
  }
  
  public BackgroundFrequencyVectorSparsifier(Vector collectionFreqsVector, float minCfRatio) {
    setCollectionFrequencies(collectionFreqsVector);
    setMinCfRatio(minCfRatio);
  }
  
  public BackgroundFrequencyVectorSparsifier(double[] collectionFrequencies, float minCfRatio) {
    this(new DenseVector(collectionFrequencies), minCfRatio);
  }
  
  private void setMinCfRatio(float minCfRatio) {
    this.minCfRatio = minCfRatio;
  }
  
  private void setCollectionFrequencies(Vector collectionFreqsVector) {
    collectionFrequencies = new double[collectionFreqsVector.size()];
    for (int i = 0; i < collectionFreqsVector.size(); i++) {
      collectionFrequencies[i] = collectionFreqsVector.get(i);
      corpusWeight += collectionFrequencies[i];
    }
  }

  @Override
  public void initialize() throws IOException {
    Configuration conf = getConf();
    if (conf == null || !Float.isNaN(minCfRatio)) {
      throw new IOException("Cannot initialize before setting Configuration object");
    }
    CVBConfig config = new CVBConfig().read(conf);
    minCfRatio = config.getCfSparsificationThreshold();
    double[] collectionFrequencies = new double[config.getNumTerms()];

    for (FileStatus stat : FileSystem.get(conf).globStatus(config.getCollectionFrequencyPath(),
                                                           PathFilters.partFilter())) {
      try {
        Iterable<String> lines = new FileLineIterable(HadoopUtil.openStream(stat.getPath(), conf));
        for (String line : lines) {
          String[] split = line.split("\t");
          int featureId = Integer.parseInt(split[0]);
          double cf = Double.parseDouble(split[1]);
          collectionFrequencies[featureId] = cf;
        }
      } catch (Exception e) {
        throw new IOException("Could not configure VectorSparsifier", e);
      }
    }
    setCollectionFrequencies(new DenseVector(collectionFrequencies));
  }
  
  @Override
  public Vector sparsify(Vector input) {
    Preconditions.checkState(collectionFrequencies != null,
                             "Collection frequencies not loaded, call initialize first");
    Vector output = new RandomAccessSparseVector(input.size());
    double mult = minCfRatio * input.norm(1) / corpusWeight;
    Iterator<Vector.Element> it = input.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      if(e.get() > collectionFrequencies[e.index()] * mult) {
        output.set(e.index(), e.get());
      }
    }
    return new SequentialAccessSparseVector(output);
  }

  /**
   * take in SparseRowMatrix, spit out SparseColumnMatrix
   * @param modelMatrix which is a SparseRowMatrix, rows keyed on topicId
   * @return rebalanced SparseMatrix, with sparse rows now keyed on featureId
   */
  @Override
  public Matrix rebalance(Matrix modelMatrix) {
    Preconditions.checkState(collectionFrequencies == null
                             || collectionFrequencies.length == modelMatrix.columnSize(),
                             "Collection frequency array not equal to feature width of the model!");
    SparseMatrix outputMatrix = new SparseMatrix(modelMatrix.numCols(),
                                                 modelMatrix.numRows());
    for(int feature = 0; feature < collectionFrequencies.length; feature++) {
      double modelFeatureWeight = 0;
      double collectionFrequency = collectionFrequencies[feature];
      for(int topic = 0; topic < modelMatrix.rowSize(); topic++) {
        modelFeatureWeight += modelMatrix.get(topic, feature);
      }
      Vector sparseOutputColumn = outputMatrix.viewRow(feature);
      if(modelFeatureWeight == 0) {
  //      if(collector != null) {
  //        collector.incrementCount(CVB0Driver.Counters.COMPLETELY_SPARSIFIED_FEATURES, 1);
  //      }
        double flatCount = collectionFrequency / modelMatrix.numRows();
        for(int topic = 0; topic < modelMatrix.rowSize(); topic++) {
          sparseOutputColumn.set(topic, flatCount);
        }
      } else {
        double mult = collectionFrequency / modelFeatureWeight;
        sparseOutputColumn.assign(Functions.mult(mult));
      }
    }
    return modelMatrix;
  }
}
