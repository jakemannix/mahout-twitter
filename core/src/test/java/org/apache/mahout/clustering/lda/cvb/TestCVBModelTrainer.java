/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class TestCVBModelTrainer extends MahoutTestCase {

  private static final float ETA = 0.1f;
  private static final float ALPHA = 0.1f;

  private String[] terms;
  private Matrix matrix;
  private Matrix sampledCorpus;
  private int numGeneratingTopics = 5;
  private int numTerms = 30;
  private Path sampleCorpusPath;

  @Before
  public void setup() throws IOException {
    matrix = ClusteringTestUtils.randomStructuredModel(numGeneratingTopics, numTerms, new DoubleFunction() {
      @Override
      public double apply(double d) {
        return 1.0 / Math.pow(d + 1.0, 3);
      }
    });

    int numDocs = 500;
    int numSamples = 10;
    int numTopicsPerDoc = 1;

    sampledCorpus = ClusteringTestUtils.sampledCorpus(matrix, RandomUtils.getRandom(1234),
                                                             numDocs, numSamples, numTopicsPerDoc);

    sampleCorpusPath = getTestTempDirPath("corpus");
    MatrixUtils.write(sampleCorpusPath, new Configuration(), sampledCorpus);
  }

  @Test
  public void testInMemoryCVB0() throws Exception {
    String[] terms = new String[26];
    for(int i=0; i<terms.length; i++) {
      terms[i] = String.valueOf((char) (i + 'a'));
    }
    int numGeneratingTopics = 3;
    int numTerms = 26;
    Matrix matrix = ClusteringTestUtils.randomStructuredModel(numGeneratingTopics, numTerms, new DoubleFunction() {
      @Override public double apply(double d) {
        return 1.0 / Math.pow(d + 1.0, 2);
      }
    });

    int numDocs = 100;
    int numSamples = 20;
    int numTopicsPerDoc = 1;

    Matrix sampledCorpus = ClusteringTestUtils.sampledCorpus(matrix, RandomUtils.getRandom(),
                                                             numDocs, numSamples, numTopicsPerDoc);

    List<Double> perplexities = Lists.newArrayList();
    int numTrials = 1;
    for (int numTestTopics = 1; numTestTopics < 2 * numGeneratingTopics; numTestTopics++) {
      double[] perps = new double[numTrials];
      for(int trial = 0; trial < numTrials; trial++) {
        InMemoryCollapsedVariationalBayes0 cvb =
          new InMemoryCollapsedVariationalBayes0(sampledCorpus, terms, numTestTopics, ALPHA, ETA,
                                                 2, 1, 0, (trial+1) * 123456L);
        cvb.setVerbose(true);
        perps[trial] = cvb.iterateUntilConvergence(0, 5, 0, 0.2);
        System.out.println(perps[trial]);
      }
      Arrays.sort(perps);
      System.out.println(Arrays.toString(perps));
      perplexities.add(perps[0]);
    }
    System.out.println(Joiner.on(",").join(perplexities));
  }

  @Test
  public void testRandomStructuredModelViaMR() throws Exception {
    int numIterations = 10;
    List<Double> perplexities = Lists.newArrayList();
    int startTopic = numGeneratingTopics - 1;
    int numTestTopics = startTopic;
    while(numTestTopics < numGeneratingTopics + 2) {
      Path topicModelStateTempPath = getTestTempDirPath("topicTemp" + numTestTopics);
      Configuration conf = new Configuration();
      CVBConfig cvbConfig = new CVBConfig().setAlpha(ALPHA).setEta(ETA).setNumTopics(numTestTopics)
          .setBackfillPerplexity(false).setConvergenceDelta(0).setDictionaryPath(null)
          .setModelTempPath(topicModelStateTempPath).setTestFraction(0.2f).setNumTerms(numTerms)
          .setMaxIterations(numIterations).setInputPath(sampleCorpusPath).setNumTrainThreads(1)
          .setNumUpdateThreads(1).setIterationBlockSize(1);
      CVB0Driver.run(conf, cvbConfig);
      perplexities.add(lowestPerplexity(conf, topicModelStateTempPath));
      numTestTopics++;
    }
    int bestTopic = -1;
    double lowestPerplexity = Double.MAX_VALUE;
    for(int t = 0; t < perplexities.size(); t++) {
      if(perplexities.get(t) < lowestPerplexity) {
        lowestPerplexity = perplexities.get(t);
        bestTopic = t + startTopic;
      }
    }
    assertEquals("The optimal number of topics is not that of the generating distribution",
        numGeneratingTopics, bestTopic);
    System.out.println("Perplexities: " + Joiner.on(", ").join(perplexities));
  }


  @Test
  public void testRandomStructuredModelWithDocTopicPriorPersistence() throws Exception {
    int numIterations = 20;
    List<Double> perplexities = Lists.newArrayList();
    int startTopic = numGeneratingTopics - 1;
    int numTestTopics = startTopic;
    while(numTestTopics < numGeneratingTopics + 2) {
      Path topicModelStateTempPath = getTestTempDirPath("topicTemp" + numTestTopics);
      Configuration conf = new Configuration();
      CVBConfig cvbConfig = new CVBConfig().setAlpha(ALPHA).setEta(ETA).setNumTopics(numTestTopics)
                                .setBackfillPerplexity(false).setConvergenceDelta(0).setDictionaryPath(null)
                                .setModelTempPath(topicModelStateTempPath).setTestFraction(0.2f).setNumTerms(numTerms)
                                .setMaxIterations(numIterations).setInputPath(sampleCorpusPath).setNumTrainThreads(1)
                                .setNumUpdateThreads(1).setIterationBlockSize(1).setPersistDocTopics(true);
      CVB0Driver.run(conf, cvbConfig);
      perplexities.add(lowestPerplexity(conf, topicModelStateTempPath));
      numTestTopics++;
    }
    int bestTopic = -1;
    double lowestPerplexity = Double.MAX_VALUE;
    for(int t = 0; t < perplexities.size(); t++) {
      if(perplexities.get(t) < lowestPerplexity) {
        lowestPerplexity = perplexities.get(t);
        bestTopic = t + startTopic;
      }
    }
    assertEquals("The optimal number of topics is not that of the generating distribution",
                    numGeneratingTopics, bestTopic);
    System.out.println("Perplexities: " + Joiner.on(", ").join(perplexities));
  }


  @Test
  public void testPriorDocTopics() throws Exception {
    sampledCorpus.numRows();
    Matrix sampledCorpusPriors = new DenseMatrix(sampledCorpus.numRows(), numGeneratingTopics);
    for(int docId = 0; docId < sampledCorpus.numRows(); docId++) {
      Vector doc = sampledCorpus.viewRow(docId);
      int term = mostProminentFeature(doc);
      Vector prior = new DenseVector(numGeneratingTopics);
      prior.assign(1.0/numGeneratingTopics);
      if(term % (numTerms / numGeneratingTopics) == 0) {
        int topic = expectedTopicForTerm(matrix, term);
        //prior.set(numGeneratingTopics - (term/numGeneratingTopics) - 1, 1);
        prior.set(topic, 1);
        prior = prior.normalize(1);
      }
      sampledCorpusPriors.assignRow(docId, prior);
    }
    Path priorPath = getTestTempDirPath("prior");
    Configuration conf = new Configuration();
    MatrixUtils.write(priorPath, conf, sampledCorpusPriors);
    Path topicModelStateTempPath = getTestTempDirPath("topicTemp");
    Path outputPath = new Path(getTestTempDirPath(), "finalOutput");
    int numIterations = 10;
    CVBConfig cvbConfig = new CVBConfig().setAlpha(ALPHA).setEta(ETA).setNumTopics(numGeneratingTopics)
          .setBackfillPerplexity(false).setConvergenceDelta(0).setDictionaryPath(null)
          .setModelTempPath(topicModelStateTempPath).setTestFraction(0.2f).setNumTerms(numTerms)
          .setMaxIterations(numIterations).setInputPath(sampleCorpusPath).setNumTrainThreads(1)
          .setDocTopicPriorPath(priorPath)
          .setNumUpdateThreads(1).setIterationBlockSize(1).setOutputPath(outputPath);
    CVB0Driver.run(conf, cvbConfig);
    double perplexity = lowestPerplexity(conf, topicModelStateTempPath);
    System.out.println("Perplexity: " + perplexity);
    List<Path> modelParts = Lists.newArrayList();
    Path p = new Path(topicModelStateTempPath, "model-" + numIterations);
    for(FileStatus fileStatus : p.getFileSystem(conf).listStatus(p, PathFilters.partFilter())) {
      modelParts.add(fileStatus.getPath());
    }
    Pair<Matrix, Vector> model = TopicModel.loadModel(conf, modelParts.toArray(new Path[0]));
    for(int topic = 0; topic < numGeneratingTopics; topic++) {
      Vector topicDist = model.getFirst().viewRow(topic);
      int term = mostProminentFeature(topicDist);
      int expectedTopicForTerm = expectedTopicForTerm(matrix, term);
      System.out.println("Expecting that term " + term + " was from topic " + expectedTopicForTerm
                         + " we got: " + topic);
      assertEquals(expectedTopicForTerm, topic);
    }
  }


  @Test
  public void testInference() throws Exception {
    String[] terms = new String[26];
    for(int i=0; i<terms.length; i++) {
      terms[i] = String.valueOf((char) (i + 'a'));
    }
    int numGeneratingTopics = 5;
    int numTerms = 26;
    Matrix matrix = ClusteringTestUtils.randomStructuredModel(numGeneratingTopics, numTerms, new DoubleFunction() {
      @Override public double apply(double d) {
        return 1.0 / Math.pow(d + 1.0, 2);
      }
    });

    int numDocs = 100;
    int numSamples = 50;
    int numTopicsPerDoc = 2;

    Matrix sampledCorpus = ClusteringTestUtils.sampledCorpus(matrix, RandomUtils.getRandom(),
        numDocs, numSamples, numTopicsPerDoc);

    TopicModel model = new TopicModel(matrix, ETA, ALPHA, terms, 1, 1.0);

    Vector doc = sampledCorpus.viewRow(0);
    Vector docTopicPrior = new DenseVector(numGeneratingTopics).assign(1.0 / numGeneratingTopics);

    Vector docTopicInference = model.infer(doc, docTopicPrior, EPSILON, 100);

    double perplexity = Double.MAX_VALUE;
    int numIters = 0;
    while(numIters++ < 100) {
      double oldPerplexity = perplexity;
      model.trainDocTopicModel(doc, docTopicPrior, new SparseRowMatrix(numGeneratingTopics, numTerms));
      perplexity = model.perplexity(doc, docTopicPrior);
      if(oldPerplexity < perplexity) {
        fail("Perplexity should not increase: old[" + oldPerplexity + "], new[" + perplexity + "]");
      }
      if(oldPerplexity - perplexity < EPSILON) {
        break;
      }
    }
    if(numIters > 50) {
      fail("Too many iterations: " + numIters);
    } else {
      System.out.println("Converged in: " + numIters + " iterations to perplexity: " + perplexity);
    }
  }

  @Test
  public void testInferencePerformance() throws Exception {
    Configuration conf = new Configuration();
    Path modelPath = new Path("/Users/jake/lda_2012_01_09_16/part-*");
    FileStatus[] fs = FileSystem.get(conf).globStatus(modelPath);
    Path[] paths = new Path[fs.length];
    for(int i=0; i<fs.length; i++) {
      paths[i] = fs[i].getPath();
    }
    TopicModel model = new TopicModel(conf, ETA, ALPHA, null, 1, 1.0, paths);
    Path dictPath = new Path("/Users/jake/dict_2012_01_09_16_1");
    DocumentVectorizer vectorizer = new InMemoryDictionaryVectorizer(conf, dictPath, null,
        new StandardAnalyzer(Version.LUCENE_31));
    SequenceFileIterable<IntWritable, VectorWritable> docs = new SequenceFileIterable<IntWritable, VectorWritable>(
        new Path("/Users/jake/userdoc_matrix_2012_01_09_16_part_00000"), conf
    );
    int numDocs = 0;
    int numNonZeroes = 0;
    long timeSpent = 0;
    Vector uniformPrior = new DenseVector(model.getNumTopics()).assign(1.0 / model.getNumTopics());
    for(Pair<IntWritable, VectorWritable> docPair : docs) {
      long start = System.nanoTime();
      Vector docTopics = model.infer(docPair.getSecond().get(), uniformPrior.clone(), 1e-6, 50);
      timeSpent += (System.nanoTime() - start);
      numDocs++;
      numNonZeroes += docPair.getSecond().get().getNumNondefaultElements();
      if(numDocs % 100 == 0) {
        System.out.println((timeSpent / (numDocs * 1e6)) + "ms/doc (" + (numNonZeroes / numDocs) + " nonzeroes/doc)");
      }
    }
    double millis = timeSpent / 1e6;
    System.out.println("Took " + millis + "ms to compute " + numDocs + " inferences of " + (numNonZeroes / numDocs) +
        " nonzero elements avg each");

  }


  /*
  private int expectedTopicForTerm(int term) {
    return ((term / (numTerms / numGeneratingTopics)) + 2) % numGeneratingTopics;
  }
  */

  private int expectedTopicForTerm(Matrix model, int term) {
    return mostProminentFeature(model.viewColumn(term));
  }

  private int mostProminentFeature(Vector doc) {
    int term = -1;
    double maxVal = Double.NEGATIVE_INFINITY;
    for(Vector.Element e : doc) {
      if(Math.abs(e.get()) > maxVal) {
        maxVal = Math.abs(e.get());
        term = e.index();
      }
    }
    return term;
  }

  private static double lowestPerplexity(Configuration conf, Path topicModelTemp)
      throws IOException {
    double lowest = Double.MAX_VALUE;
    double current;
    int iteration = 2;
    while(!Double.isNaN(current = CVB0Driver.readPerplexity(conf, topicModelTemp, iteration))) {
      lowest = Math.min(current, lowest);
      iteration++;
    }
    return lowest;
  }

}
