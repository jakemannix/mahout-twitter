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

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ThreadFactoryBuilder;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.stats.Sampler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Thin wrapper around a {@link Matrix} of counts of occurrences of (topic, term) pairs.  Dividing
 * {code topicTermCount.viewRow(topic).get(term)} by the sum over the values for all terms in that
 * row yields p(term | topic).  Instead dividing it by all topic columns for that term yields
 * p(topic | term).
 *
 * Multithreading is enabled for the {@code update(Matrix)} method: this method is async, and
 * merely submits the matrix to a work queue.  When all work has been submitted,
 * {@code awaitTermination()} should be called, which will block until updates have been
 * accumulated.
 */
public class TopicModel extends TopicModelBase implements Iterable<MatrixSlice> {
  private static final Logger log = LoggerFactory.getLogger(TopicModel.class);
  private final Matrix topicTermCounts;
  private final Vector topicSums;
  private final int numTerms;
  private final double eta;
  private final double alpha;
  private final Vector uniform;
  private final Sampler sampler;
  private final int numThreads;
  private final Updater[] updaters;

  public TopicModel(Configuration conf, double eta, double alpha,
      int numThreads, double modelWeight, Path... modelpath) throws IOException {
    this(loadModel(conf, modelpath), eta, alpha, numThreads, modelWeight);
    setConf(conf);
  }

  public TopicModel(int numTopics, int numTerms, double eta, double alpha,
      int numThreads, double modelWeight) {
    this(new DenseMatrix(numTopics, numTerms), new DenseVector(numTopics), eta, alpha,
        numThreads, modelWeight);
  }

  private TopicModel(Pair<Matrix, Vector> model, double eta, double alpha,
      int numThreads, double modelWeight) {
    this(model.getFirst(), model.getSecond(), eta, alpha, numThreads, modelWeight);
  }

  public TopicModel(Matrix topicTermCounts, Vector topicSums, double eta, double alpha,
    double modelWeight) {
    this(topicTermCounts, topicSums, eta, alpha, 1, modelWeight);
  }

  public TopicModel(Matrix topicTermCounts, double eta, double alpha,
      int numThreads, double modelWeight) {
    this(topicTermCounts, viewRowSums(topicTermCounts),
        eta, alpha, numThreads, modelWeight);
  }

  public TopicModel(Matrix topicTermCounts, Vector topicSums, double eta, double alpha,
    int numThreads, double modelWeight) {
    super(topicSums.size());
    this.topicTermCounts = topicTermCounts;
    this.topicSums = topicSums;
    this.numTerms = topicTermCounts.numCols();
    this.eta = eta;
    this.alpha = alpha;
    this.uniform = new DenseVector(numTopics).assign(1.0 / numTopics);
    this.sampler = new Sampler(RandomUtils.getRandom());
    this.numThreads = numThreads;
    this.updaters = new Updater[numThreads];
    if(modelWeight != 1) {
      topicSums.assign(Functions.mult(modelWeight));
      for(int x = 0; x < numTopics; x++) {
        topicTermCounts.viewRow(x).assign(Functions.mult(modelWeight));
      }
    }
    initializeThreadPool();
  }

  public Matrix getTopicVectors() {
    return topicTermCounts;
  }

  private static Vector viewRowSums(Matrix m) {
    Vector v = new DenseVector(m.numRows());
    for(MatrixSlice slice : m) {
      v.set(slice.index(), slice.vector().norm(1));
    }
    return v;
  }

  private void initializeThreadPool() {
    ThreadPoolExecutor threadPool =
        new ThreadPoolExecutor(numThreads, numThreads, 0, TimeUnit.SECONDS,
                               new ArrayBlockingQueue<Runnable>(numThreads * 10),
                               new ThreadFactoryBuilder().setDaemon(false).build());
    threadPool.allowCoreThreadTimeOut(false);
    for(int i = 0; i < numThreads; i++) {
      updaters[i] = new Updater();
      threadPool.submit(updaters[i]);
    }
  }

  @Override
  public Iterator<MatrixSlice> iterator() {
    return topicTermCounts.iterateAll();
  }

  public int sampleTerm(Vector topicDistribution) {
    return sampler.sample(topicTermCounts.viewRow(sampler.sample(topicDistribution)));
  }

  public int sampleTerm(int topic) {
    return sampler.sample(topicTermCounts.viewRow(topic));
  }

  public void reset() {
    for(int x = 0; x < numTopics; x++) {
      topicTermCounts.assignRow(x, new SequentialAccessSparseVector(numTerms));
    }
    topicSums.assign(1.0);
    initializeThreadPool();
  }

  public void awaitTermination() {
    for(Updater updater : updaters) {
      updater.shutdown();
    }
  }

  public void renormalize() {
    for(int x = 0; x < numTopics; x++) {
      topicTermCounts.assignRow(x, topicTermCounts.viewRow(x).normalize(1));
      topicSums.assign(1.0);
    }
  }

  @Override
  public void trainDocTopicModelSingleIteration(DocTrainingState state) {
    Vector original = state.getDocument();
    Vector topics = state.getDocTopics();
    Matrix docTopicModel = state.getDocTopicModel();
    // first calculate p(topic|term,document) for all terms in original, and all topics,
    // using p(term|topic) and p(topic|doc)
    pTopicGivenTerm(original, topics, docTopicModel);
    normalizeByTopic(original, docTopicModel);
    // now multiply, term-by-term, by the document, to get the weighted distribution of
    // term-topic pairs from this document.
    Iterator<Vector.Element> it = original.iterateNonZero();
    Vector.Element e = null;
    while(it.hasNext() && (e = it.next())!= null && e.index() < numTerms) { // protect vs. big docs
      for(int x = 0; x < numTopics; x++) {
        Vector docTopicModelColumn = docTopicModel.viewColumn(x);
        docTopicModelColumn.setQuick(e.index(), docTopicModelColumn.getQuick(e.index()) * e.get());
      }
    }
    // now recalculate p(topic|doc) by summing contributions from all of pTopicGivenTerm
    topics.assign(0.0);
    for(int x = 0; x < numTopics; x++) {
      it = original.iterateNonZero();
      double norm = 0;
      while(it.hasNext() && (e = it.next())!= null && e.index() < numTerms) {
        norm += docTopicModel.get(e.index(), x);
      }
      topics.set(x, norm);
    }
    // now renormalize so that sum_x(p(x|doc)) = 1
    topics.assign(Functions.mult(1.0 / topics.norm(1)));
  }

  public Vector expectedTermCounts(Vector original, Vector docTopics) {
    Vector pTerm = original.like();
    Iterator<Vector.Element> it = original.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int term = e.index();
      // p(a) = sum_x (p(a|x) * p(x|i))
      double pA = 0;
      for(int x = 0; x < numTopics; x++) {
        pA += (topicTermCounts.viewRow(x).get(term) / topicSums.get(x)) * docTopics.get(x);
      }
      pTerm.set(term, pA);
    }
    return pTerm;
  }

  @Override
  public Vector infer(DocTrainingState state, double minRelPerplexityDiff, int maxIters) {
    Vector docTopic = state.getDocTopics().clone();
    Vector original = state.getDocument();
    double oldPerplexity;
    double perplexity = Double.MAX_VALUE;
    double relPerplexityDiff = Double.MAX_VALUE;
    int iter = 1;
    for (; iter <= maxIters && relPerplexityDiff > minRelPerplexityDiff; ++iter) {
      oldPerplexity = perplexity;
      perplexity = perplexity(original, docTopic);
      state.setPerplexity(perplexity);
      trainDocTopicModel(state);
      if (oldPerplexity < perplexity) {
        log.warn("Document inference lead to increasing perplexity after {} iterations", iter);
        break;
      }
      relPerplexityDiff = (oldPerplexity - perplexity) / oldPerplexity;
    }
    log.debug("Relative perplexity difference of {} achieved after {} iterations", relPerplexityDiff, iter);
    return docTopic;
  }

  @Override
  public void update(DocTrainingState state) {
    for(int x = 0; x < numTopics; x++) {
      updaters[x % updaters.length].update(x, state.getDocTopicModel().viewColumn(x));
    }
  }

  public void updateTopic(int topic, Vector docTopicCounts) {
    Iterator<Vector.Element> it = docTopicCounts.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      topicTermCounts.set(topic, e.index(), topicTermCounts.get(topic, e.index()) + e.get());
    }
    topicSums.set(topic, topicSums.get(topic) + docTopicCounts.norm(1));
  }

  public void update(int termId, Vector topicCounts) {
    for(int x = 0; x < numTopics; x++) {
      Vector v = topicTermCounts.viewRow(x);
      v.set(termId, v.get(termId) + topicCounts.get(x));
    }
    topicSums.assign(topicCounts, Functions.PLUS);
  }

  /**
   * Computes {@code p(topic x|term a, document i)} distributions given input document {@code i}.
   * {@code pTGT[x][a]} is the (un-normalized) {@code p(x|a,i)}, or if docTopics is {@code null},
   * {@code p(a|x)} (also un-normalized).
   *
   * @param document doc-term vector encoding {@code w(term a|document i)}.
   * @param docTopics {@code docTopics[x]} is the overall weight of topic {@code x} in given
   *          document. If {@code null}, a topic weight of {@code 1.0} is used for all topics.
   * @param termTopicDist storage for output {@code p(x|a,i)} distributions.
   */
  private void pTopicGivenTerm(Vector document, Vector docTopics, Matrix termTopicDist) {
    // for each topic x
    for(int x = 0; x < numTopics; x++) {
      // get p(topic x | document i), or 1.0 if docTopics is null
      double topicWeight = docTopics == null ? 1.0 : docTopics.get(x);
      // get w(term a | topic x)
      Vector topicTermRow = topicTermCounts.viewRow(x);
      // get \sum_a w(term a | topic x)
      double topicSum = topicSums.get(x);
      // get p(topic x | term a) distribution to update
      Vector termTopicRow = termTopicDist.viewColumn(x);
      // cache factor which is the same for all terms, for this fixed topic.
      double topicMult = (topicWeight + alpha) / (topicSum + eta * numTerms);

      // for each term a in document i with non-zero weight
      Iterator<Vector.Element> it = document.iterateNonZero();
      Vector.Element e = null;
      while(it.hasNext() && (e = it.next()) != null && e.index() < numTerms) {
        int termIndex = e.index();

        // calc un-normalized p(topic x | term a, document i)
        double termTopicLikelihood = (topicTermRow.get(termIndex) + eta) * topicMult;
        termTopicRow.set(termIndex, termTopicLikelihood);
      }
    }
  }

  /**
   * sum_x sum_a (c_ai * log(p(x|i) * p(a|x)))
   */
  @Override
  public double perplexity(Vector document, Vector docTopics) {
    double perplexity = 0;
    double norm = docTopics.norm(1) + (docTopics.size() * alpha);
    Iterator<Vector.Element> it = document.iterateNonZero();
    Vector.Element e = null;
    while(it.hasNext() && (e = it.next()) != null && e.index() < numTerms) {
      int term = e.index();
      double prob = 0;
      for(int x = 0; x < numTopics; x++) {
        double d = (docTopics.get(x) + alpha) / norm;
        double p = d * (topicTermCounts.viewRow(x).get(term) + eta)
                   / (topicSums.get(x) + eta * numTerms);
        prob += p;
      }
      perplexity += e.get() * Math.log(prob);
    }
    return -perplexity;
  }

  /**
   *
   * @param doc just here to provide a sparse iterator
   * @param perTopicSparseDistributions
   */
  private void normalizeByTopic(Vector doc, Matrix perTopicSparseDistributions) {
    Iterator<Vector.Element> it = doc.iterateNonZero();
    // then make sure that each of these is properly normalized by topic: sum_x(p(x|t,d)) = 1
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int a = e.index();
      double sum = 0;
      for(int x = 0; x < numTopics; x++) {
        sum += perTopicSparseDistributions.get(a, x);
      }
      for(int x = 0; x < numTopics; x++) {
        perTopicSparseDistributions.set(a, x, perTopicSparseDistributions.get(a, x) / sum);
      }
    }
  }

  private final class Updater implements Runnable {
    private final ArrayBlockingQueue<Pair<Integer, Vector>> queue =
        new ArrayBlockingQueue<Pair<Integer, Vector>>(100);
    private boolean shutdown = false;
    private boolean shutdownComplete = false;

    public void shutdown() {
      try {
        synchronized (this) {
          while(!shutdownComplete) {
            shutdown = true;
            wait(10000L); // Arbitrarily, wait 10 seconds rather than forever for this
          }
        }
      } catch (InterruptedException e) {
        log.warn("Interrupted waiting to shutdown() : ", e);
      }
    }

    public boolean update(int topic, Vector v) {
      if(shutdown) { // maybe don't do this?
        throw new IllegalStateException("In SHUTDOWN state: cannot submit tasks");
      }
      while(true) { // keep trying if interrupted
        try {
          // start async operation by submitting to the queue
          queue.put(Pair.of(topic, v));
          // return once you got access to the queue
          return true;
        } catch (InterruptedException e) {
          log.warn("Interrupted trying to queue update:", e);
        }
      }
    }

    @Override public void run() {
      while(!shutdown) {
        try {
          Pair<Integer, Vector> pair = queue.poll(1, TimeUnit.SECONDS);
          if(pair != null) {
            updateTopic(pair.getFirst(), pair.getSecond());
          }
        } catch (InterruptedException e) {
          log.warn("Interrupted waiting to poll for update", e);
        }
      }
      // in shutdown mode, finish remaining tasks!
      for(Pair<Integer, Vector> pair : queue) {
        updateTopic(pair.getFirst(), pair.getSecond());
      }
      synchronized (this) {
        shutdownComplete = true;
        notifyAll();
      }
    }
  }

}
