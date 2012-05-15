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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;

/**
 * Run ensemble learning via loading the two {@link TopicModel} instances:
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
 */
public class CachingCVB0Mapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
  private static final Logger log = LoggerFactory.getLogger(CachingCVB0Mapper.class);
  protected CVBConfig config;
  protected TopicModelBase readModel;
  protected TopicModelBase writeModel;
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
    maxIters = config.getMaxItersPerDoc();
    float modelWeight = config.getModelWeight();

    if (config.getCollectionFrequencyPath() != null) {
      Class<? extends VectorSparsifier> sparsifierClass = NoopVectorSparsifier.class;
/*          context.getConfiguration().getClass(SparsifyingVectorSumReducer.SPARSIFIER_CLASS,
                                             NoopVectorSparsifier.class,
                                             VectorSparsifier.class); */
      VectorSparsifier bfvs = new BackgroundFrequencyVectorSparsifier();
      bfvs.setConf(conf);
      bfvs.initialize();
      sparsifier = bfvs;
    }

    log.info("Initializing read model");
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
      
      readModel = new SparseTopicModel(modelMatrix, eta, alpha, numUpdateThreads, modelWeight);
    } else {
      log.info("No model files found");
      throw new IOException("No model files found, must pre-initialize model somehow");
    }

    log.info("Initializing write model");
    // TODO: using "modelWeight == 1" as the switch here is BAD. Online LDA with modelWeight 1 is ok
    writeModel = modelWeight == 1
        ? new SparseTopicModel(numTopics, numTerms, eta, alpha, numUpdateThreads, 1)
        : readModel;
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
      throws IOException, InterruptedException {
    context.getCounter(CVB0Driver.Counters.SAMPLED_DOCUMENTS).increment(1); // TODO: rename me
    DocTrainingState state = new DocTrainingState.Builder()
                                 .setDocument(document.get())
                                 .setNumTopics(numTopics)
                                 .setMaxIters(maxIters).build();
    readModel.trainDocTopicModel(state);
//    for(int topic = 0; topic < numTopics; topic++) {
//      context.write(new IntWritable(topic),
//                    new VectorWritable(state.getDocTopicModel().viewColumn(topic)));
//    }
    writeModel.update(state);
    if(shouldFlush(writeModel.getNumNonZeroes())) {
      flush(context);
    }
  }

  @Override
  protected void cleanup(final Context context) throws IOException, InterruptedException {
    log.info("Writing model");
    flush(context);
  }

  private void flush(Context context) throws IOException, InterruptedException {
    writeModel.awaitTermination();
    for(MatrixSlice topic : writeModel.getTopicVectors()) {
      context.write(new IntWritable(topic.index()), new VectorWritable(topic.vector()));
    }
    writeModel = new SparseTopicModel(numTopics, config.getNumTerms(),
                                         config.getEta(), config.getAlpha(),
                                         config.getNumUpdateThreads(), 1);
  }

  private boolean shouldFlush(int numCurrentNonZeroes) {
    Runtime runtime = Runtime.getRuntime();
    long freeMem = runtime.freeMemory();
    long totalMem = runtime.totalMemory();
    return freeMem / totalMem > 0.25;
  }
}
