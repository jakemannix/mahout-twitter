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

import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.io.Files;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.lda.LDASampler;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.MatrixUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class TestCVBModelTrainer extends MahoutTestCase {

  private static final float ETA = 0.1f;
  private static final float ALPHA = 0.1f;

  private String[] terms;
  private Matrix matrix;
  private Matrix sampledCorpus;
  private Vector collectionFrequencies;
  private int numGeneratingTopics = 5;
  private int numTerms = 30;
  private Path sampleCorpusPath;
  private Path testCorpusPath;
  private Path sampleCorpusCollectionFrequencyPath;

  @Before
  public void setup() throws IOException {
    Configuration conf = new Configuration();
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
    MatrixUtils.write(sampleCorpusPath, conf, sampledCorpus);
    
    Matrix testSet = ClusteringTestUtils.sampledCorpus(matrix, RandomUtils.getRandom(9876),
                                                       numDocs / 10, numSamples, numTopicsPerDoc);

    testCorpusPath = getTestTempDirPath("testSet");

    MatrixUtils.write(testCorpusPath, conf, testSet);
    
    sampleCorpusCollectionFrequencyPath = getTestTempDirPath("collectionFrequencies");
    collectionFrequencies = new DenseVector(matrix.columnSize());
    for(int feature = 0; feature < collectionFrequencies.size(); feature++) {
      double fCf = 0;
      for(int docId = 0; docId < sampledCorpus.rowSize(); docId++) {
        fCf += sampledCorpus.get(docId, feature);
      }
      collectionFrequencies.set(feature, fCf);
    }
    File cfFile = new File(sampleCorpusCollectionFrequencyPath.toUri());
    cfFile.delete();
    cfFile.createNewFile();
    for(int i = 0; i < collectionFrequencies.size(); i++) {
      Files.append("" + i + "\t" + collectionFrequencies.get(i), cfFile, Charsets.UTF_8);
    }
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


  private void runEndToEndMRTest(CVBConfig cvbConfig) throws Exception {
    runEndToEndMRTest(cvbConfig, 1, 1);
  }

  /**
   *
   * @param cvbConfig the training configuration
   * @param numSmallerTopics start at numGeneratingTopics - this number
   * @param numLargerTopics end at numGeneratingTopics + this number
   * @throws Exception usually IOException
   */
  private void runEndToEndMRTest(CVBConfig cvbConfig,
                                 int numSmallerTopics,
                                 int numLargerTopics) throws Exception {
    List<Double> perplexities = Lists.newArrayList();
    int startTopic = numGeneratingTopics - numSmallerTopics;
    int numTestTopics = startTopic;
    while(numTestTopics <= numGeneratingTopics + numLargerTopics) {
      Path topicModelStateTempPath = getTestTempDirPath("topicTemp" + numTestTopics);
      Configuration conf = new Configuration();
      cvbConfig.setModelTempPath(topicModelStateTempPath).setNumTopics(numTestTopics);
      CVB0Driver.run(conf, cvbConfig);
      Pair<Integer, Double> lowestPerplexity = lowestPerplexity(conf, topicModelStateTempPath);
      int bestIter = lowestPerplexity.getFirst();
      assertTrue("Iteration with lowest perplexity should not be this early: " + bestIter,
                  bestIter > cvbConfig.getMaxIterations() / 2);
      perplexities.add(lowestPerplexity.getSecond());
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
  public void testRandomStructuredModelViaMR() throws Exception {
    int numIterations = 10;
    CVBConfig cvbConfig = defaultConfig().setMaxIterations(numIterations);
    runEndToEndMRTest(cvbConfig);
  }


  @Test
  public void testRandomStructuredModelWithDocTopicPriorPersistence() throws Exception {
    int numIterations = 20;
    CVBConfig cvbConfig = defaultConfig().setPersistDocTopics(true).setMaxIterations(numIterations);
    runEndToEndMRTest(cvbConfig);
  }

  @Test
  public void testIterationByIterationInference() throws Exception {
    doTestIterationByIterationInference(new TopicModel(matrix, ETA, ALPHA, 1, 1.0));
    doTestIterationByIterationInference(new SparseTopicModel(matrix, ETA, ALPHA, 1, 1.0));
  }

  @Test
  public void testTrainingKeepsModelWeightConstant() throws Exception {
    doTestTrainingKeepsModelWeightConstant(new TopicModel(matrix, ETA, ALPHA, 1, 1.0),
                                           new TopicModel(matrix.numRows(), matrix.numCols(),
                                                          ETA, ALPHA, 1, 1.0));
    doTestTrainingKeepsModelWeightConstant(new SparseTopicModel(matrix, ETA, ALPHA, 1, 1.0),
                                           new SparseTopicModel(matrix.numRows(), matrix.numCols(),
                                                                ETA, ALPHA, 1, 1.0));
  }

  private void doTestTrainingKeepsModelWeightConstant(TopicModelBase readModel,
                                                      TopicModelBase writeModel) throws Exception {
    double corpusWeight = 0;
    for(MatrixSlice slice : sampledCorpus) {
      corpusWeight += slice.vector().norm(1);
      DocTrainingState state = new DocTrainingState.Builder()
                                   .setDocument(slice.vector())
                                   .setMaxIters(30)
                                   .setNumTopics(readModel.numTopics).build();
      readModel.trainDocTopicModel(state);
      writeModel.update(state);
    }
    double modelWeight = 0;
    for(MatrixSlice slice : writeModel.getTopicVectors()) {
      modelWeight += slice.vector().norm(1);
    }
    double relativeDiff = Math.abs(modelWeight - corpusWeight) / modelWeight;
    assertTrue("Model should be same weight as the corpus: " + relativeDiff,
               relativeDiff < 1e-6);
  }

  private void doTestIterationByIterationInference(TopicModelBase topicModel) throws Exception {
    Vector document = new RandomAccessSparseVector(numTerms);
    Random random = new Random(1234L);
    LDASampler modelSampler = new LDASampler(matrix, random);
    Vector genTopics = new DenseVector(matrix.numRows());
    for(int i = 0; i < 3; i++) {
      int topic = random.nextInt(genTopics.size());
      genTopics.set(topic, genTopics.get(topic) + 1);
    }
    for(int sample : modelSampler.sample(genTopics, 5)) {
      document.set(sample, document.get(sample) + 1);
    }
    
    DocTrainingState state = new DocTrainingState.Builder().setDocument(document)
                                                           .setNumTopics(numGeneratingTopics)
                                                           .setMaxIters(1)
                                                           .build();
    Vector docTopics = new DenseVector(state.getDocTopics());
    int iter = 0;
    while(iter++ < 10) {
      topicModel.trainDocTopicModel(state);
      Vector nextDocTopics = new DenseVector(state.getDocTopics());
      assertTrue("failed to get closer on iteration: " + iter,
                 genTopics.minus(nextDocTopics).norm(1) < genTopics.minus(docTopics).norm(1));
      docTopics = nextDocTopics;
    }
  }

  @Test
  @Ignore("Online LDA currently not working quite right")
  public void testModelWeightNotEqualToOne() throws Exception {
    int numIterations = 20;
    CVBConfig cvbConfig = defaultConfig().setModelWeight(1.1f)
                                         .setMaxIterations(numIterations);
    runEndToEndMRTest(cvbConfig);
  }


  private CVBConfig defaultConfig() {
    return new CVBConfig().setAlpha(ALPHA).setEta(ETA)
               .setBackfillPerplexity(false).setConvergenceDelta(0).setDictionaryPath(null)
               .setNumTerms(numTerms)
               .setInputPath(sampleCorpusPath)
               .setTestSetPath(testCorpusPath)
               .setCollectionFrequencyPath(sampleCorpusCollectionFrequencyPath)
               .setCFSparsificationThreshold(1)
               .setNumTrainThreads(1)
               .setNumUpdateThreads(1).setIterationBlockSize(1);
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
    int numTopics = numGeneratingTopics - 2;
    CVBConfig cvbConfig = new CVBConfig().setAlpha(ALPHA).setEta(ETA)
          .setNumTopics(numTopics)
          .setBackfillPerplexity(false).setConvergenceDelta(0).setDictionaryPath(null)
          .setModelTempPath(topicModelStateTempPath).setNumTerms(numTerms)
          .setMaxIterations(numIterations).setInputPath(sampleCorpusPath).setNumTrainThreads(1)
          .setDocTopicPriorPath(priorPath)
          .setNumUpdateThreads(1).setIterationBlockSize(1).setOutputPath(outputPath);
    CVB0Driver.run(conf, cvbConfig);
    Pair<Integer, Double> lowestPerplexity = lowestPerplexity(conf, topicModelStateTempPath);
    double perplexity = lowestPerplexity.getSecond();
    int bestIteration = lowestPerplexity.getFirst();
    assertTrue("Lowest perplexity should not be this early in iterations: " + bestIteration,
               bestIteration > numIterations / 2);
    System.out.println("Perplexity: " + perplexity);
    List<Path> modelParts = Lists.newArrayList();
    Path p = new Path(topicModelStateTempPath, "model-" + numIterations);
    for(FileStatus fileStatus : p.getFileSystem(conf).listStatus(p, PathFilters.partFilter())) {
      modelParts.add(fileStatus.getPath());
    }
    Pair<Matrix, Vector> model = TopicModelBase.loadModel(conf, modelParts.toArray(new Path[0]));
    for(int topic = 0; topic < numTopics; topic++) {
      Vector topicDist = model.getFirst().viewRow(topic);
      int term = mostProminentFeature(topicDist);
      int expectedTopicForTerm = expectedTopicForTerm(matrix, term);
      System.out.println("Expecting that term " + term + " was from topic " + expectedTopicForTerm
                         + " we got: " + topic);
      assertEquals(expectedTopicForTerm, topic);
    }
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

  private static Pair<Integer, Double> lowestPerplexity(Configuration conf, Path topicModelTemp)
      throws IOException {
    double lowest = Double.MAX_VALUE;
    double current;
    int iteration = 2;
    int bestIteration = -1;
    while(!Double.isNaN(current = CVB0Driver.readPerplexity(conf, topicModelTemp, iteration))) {
      if(current < lowest) {
        bestIteration = iteration;
        lowest = current;
      }
      iteration++;
    }
    return Pair.of(bestIteration, lowest);
  }

}
