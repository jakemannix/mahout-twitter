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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

import com.google.common.base.Preconditions;

/**
 * Run ensemble learning via loading the {@link ModelTrainer} with two {@link TopicModel} instances:
 * one from the previous iteration, the other empty.  Inference is done on the first, and the
 * learning updates are stored in the second, and only emitted at cleanup().
 *
 * In terms of obvious performance improvements still available, the memory footprint in this
 * Mapper could be dropped by half if we accumulated model updates onto the model we're using
 * for inference, which might also speed up convergence, as we'd be able to take advantage of
 * learning <em>during</em> iteration, not just after each one is done.  Most likely we don't
 * really need to accumulate double values in the model either, floats would most likely be
 * sufficient.  Between these two, we could squeeze another factor of 4 in memory efficiency.
 *
 * In terms of CPU, we're re-learning the p(topic|doc) distribution on every iteration, starting
 * from scratch.  This is usually only 10 fixed-point iterations per doc, but that's 10x more than
 * only 1.  To avoid having to do this, we would need to do a map-side join of the unchanging
 * corpus with the continually-improving p(topic|doc) matrix, and then emit multiple outputs
 * from the mappers to make sure we can do the reduce model averaging as well.  Tricky, but
 * possibly worth it.
 *
 * {@link ModelTrainer} already takes advantage (in maybe the not-nice way) of multi-core
 * availability by doing multithreaded learning, see that class for details.
 */
public class CachingCVB0Mapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
  private static final Logger log = LoggerFactory.getLogger(CachingCVB0Mapper.class);
  protected CVBConfig config;
  protected ModelTrainer modelTrainer;
  protected int maxIters;
  protected int numTopics;
  protected VectorSparsifier sparsifier;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    log.info("Retrieving configuration");
    Configuration conf = context.getConfiguration();
    config = new CVBConfig().read(conf);
    float eta = config.getEta();
    float alpha = config.getAlpha();
    numTopics = config.getNumTopics();
    int numTerms = config.getNumTerms();
    int numUpdateThreads = config.getNumUpdateThreads();
    int numTrainThreads = config.getNumTrainThreads();
    maxIters = config.getMaxItersPerDoc();
    float modelWeight = config.getModelWeight();

    if (config.getCollectionFrequencyPath() != null) {
      Class<? extends VectorSparsifier> sparsifierClass = BackgroundFrequencyVectorSparsifier.class;
/*          context.getConfiguration().getClass(SparsifyingVectorSumReducer.SPARSIFIER_CLASS,
                                             NoopVectorSparsifier.class,
                                             VectorSparsifier.class); */
      BackgroundFrequencyVectorSparsifier bfvs = new BackgroundFrequencyVectorSparsifier();
      bfvs.setConf(conf);
      bfvs.setSparsifiedCounter(context.getCounter(CVB0Driver.Counters.COMPLETELY_SPARSIFIED_FEATURES));
      sparsifier = bfvs;
    }

    log.info("Initializing read model");
    TopicModel readModel;
    Path[] modelPaths = CVB0Driver.getModelPaths(conf);

    if(modelPaths != null && modelPaths.length > 0) {
      Pair<Matrix, Vector> matrix = TopicModelBase.loadModel(conf, modelPaths);
      Matrix modelMatrix = matrix.getFirst();

      if (sparsifier != null) {
        modelMatrix = sparsifier.rebalance(modelMatrix);
      }

      //Pair<Matrix, Vector> ttc = TopicModelBase.loadModel(conf, modelPaths);

      Path[] prevIterModelPath = CVB0Driver.getPreviousIterationModelPaths(conf);
      if (prevIterModelPath != null) {
        // Pair<Matrix, Vector> prevTtc = TopicModelBase.loadModel(conf, prevIterModelPath);
        // TODO: subtract previousTtc * numShards from current to get just the updates.
      }
      
      readModel = new TopicModel(modelMatrix, eta, alpha, null, numUpdateThreads, modelWeight);
    } else {
      log.info("No model files found");
      throw new IOException("No model files found, must pre-initialize model somehow");
    }

    log.info("Initializing write model");
    // TODO: using "modelWeight == 1" as the switch here is BAD. Online LDA with modelWeight 1 is ok
    TopicModel writeModel = modelWeight == 1
        ? new TopicModel(numTopics, numTerms, eta, alpha, null, numUpdateThreads, 1)
        : readModel;

    log.info("Initializing model trainer");
    modelTrainer = new ModelTrainer(readModel, writeModel, numTrainThreads, numTopics, numTerms);
    modelTrainer.start();
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
      throws IOException, InterruptedException {
    context.getCounter(CVB0Driver.Counters.SAMPLED_DOCUMENTS).increment(1);
    Vector topicVector = new DenseVector(new double[numTopics]).assign(1.0 / numTopics);
    modelTrainer.train(document.get(), topicVector, true, maxIters);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    log.info("Stopping model trainer");
    modelTrainer.stop();

    log.info("Writing model");
    TopicModelBase model = modelTrainer.getReadModel();
    for(MatrixSlice topic : model.getTopicTermCounts()) {
      context.write(new IntWritable(topic.index()), new VectorWritable(topic.vector()));
    }
  }
}
