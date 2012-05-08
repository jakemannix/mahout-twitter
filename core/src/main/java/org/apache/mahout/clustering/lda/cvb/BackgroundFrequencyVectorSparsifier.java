package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.list.IntArrayList;

public class BackgroundFrequencyVectorSparsifier extends Configured implements VectorSparsifier {
  private float minCfRatio = Float.NaN;
  private double[] collectionFrequencies = null;
  private double corpusWeight = -1;
  private Counter completelySparsifiedFeatureCounter = null;

  public BackgroundFrequencyVectorSparsifier() {
  }
  
  public BackgroundFrequencyVectorSparsifier(Vector collectionFreqsVector, float minCfRatio) {
    setCollectionFrequencies(collectionFreqsVector);
    setMinCfRatio(minCfRatio);
  }
  
  public BackgroundFrequencyVectorSparsifier(double[] collectionFrequencies, float minCfRatio) {
    this(new DenseVector(collectionFrequencies), minCfRatio);
  }
  
  public void setSparsifiedCounter(Counter counter) {
    this.completelySparsifiedFeatureCounter = counter;
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
  public void setConf(Configuration conf) {
    super.setConf(conf);
    if (conf == null || !Float.isNaN(minCfRatio)) {
      return;
    }
    CVBConfig config = new CVBConfig().read(conf);
    minCfRatio = config.getCfSparsificationThreshold();
    double[] collectionFrequencies = new double[config.getNumTerms()];
    try {
      Iterable<String> lines = new FileLineIterable(HadoopUtil.openStream(
                                                    config.getCollectionFrequencyPath(), conf));
      for (String line : lines) {
        String[] split = line.split("\t");
        int featureId = Integer.parseInt(split[0]);
        double cf = Double.parseDouble(split[1]);
        collectionFrequencies[featureId] = cf;
      }
    } catch (Exception e) {
      throw new RuntimeException("Could not configure VectorSparsifier", e);
    }
    setCollectionFrequencies(new DenseVector(collectionFrequencies));
  }
  
  @Override
  public Vector sparsify(Vector input) {

    Vector output = new RandomAccessSparseVector(input.size());

 //   IntArrayList indexes = new IntArrayList(11);
 //   DoubleArrayList values = new DoubleArrayList(11);
    double mult = minCfRatio * input.norm(1) / corpusWeight;
    for(int i = 0; i < collectionFrequencies.length; i++) {
      if(input.get(i) > collectionFrequencies[i] * mult) {
 //       indexes.add(i);
 //       values.add(input.get(i));
        output.set(i, input.get(i));
      }
    }
//    return input;
//    return output;
    return new SequentialAccessSparseVector(output);
  }

  @Override
  public Matrix rebalance(Matrix modelMatrix) {
    Preconditions.checkState(collectionFrequencies == null
                             || collectionFrequencies.length == modelMatrix.columnSize(),
                             "Collection frequency array not equal to feature width of the model!");
    for(int feature = 0; feature < collectionFrequencies.length; feature++) {
      double modelFeatureWeight = 0;
      double collectionFrequency = collectionFrequencies[feature];
      for(int topic = 0; topic < modelMatrix.rowSize(); topic++) {
        modelFeatureWeight += modelMatrix.get(topic, feature);
      }
      if(modelFeatureWeight == 0) {
        if(completelySparsifiedFeatureCounter != null) {
          completelySparsifiedFeatureCounter.increment(1);
        }
        double flatCount = collectionFrequency / modelMatrix.numRows();
        for(int topic = 0; topic < modelMatrix.rowSize(); topic++) {
          modelMatrix.set(topic, feature, flatCount);
        }
      } else {
        double mult = collectionFrequency / modelFeatureWeight;
        for(int topic = 0; topic < modelMatrix.rowSize(); topic++) {
          double oldVal = modelMatrix.get(topic, feature);
          if(oldVal != 0) {
            modelMatrix.set(topic, feature, oldVal * mult);
          }
        }
      }
    }
    return modelMatrix;
  }
}
