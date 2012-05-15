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
import org.apache.mahout.common.MemoryUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

public class CachingCVB0PerplexityMapper extends
    Mapper<IntWritable, VectorWritable, DoubleWritable, DoubleWritable> {
  private static final Logger log = LoggerFactory.getLogger(CachingCVB0PerplexityMapper.class);
  private TopicModelBase readModel;
  protected VectorSparsifier sparsifier;
  private int maxIters;
  private int numTopics;
  private final DoubleWritable outKey = new DoubleWritable();
  private final DoubleWritable outValue = new DoubleWritable();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    MemoryUtil.startMemoryLogger(5000);

    log.info("Retrieving configuration");
    Configuration conf = context.getConfiguration();
    CVBConfig config = new CVBConfig().read(conf);
    float eta = config.getEta();
    float alpha = config.getAlpha();
    numTopics = config.getNumTopics();
    int numUpdateThreads = config.getNumUpdateThreads();
    maxIters = config.getMaxItersPerDoc();
    float modelWeight = config.getModelWeight();

    log.info("Initializing read model");
    Path[] modelPaths = CVB0Driver.getModelPaths(conf);

    Class<? extends VectorSparsifier> sparsifierClass = BackgroundFrequencyVectorSparsifier.class;
/*          context.getConfiguration().getClass(SparsifyingVectorSumReducer.SPARSIFIER_CLASS,
                                              NoopVectorSparsifier.class,
                                              VectorSparsifier.class); */
    sparsifier = ReflectionUtils.newInstance(sparsifierClass, context.getConfiguration());
    sparsifier.setConf(conf);
    sparsifier.initialize();

    if(modelPaths != null && modelPaths.length > 0) {
      Pair<Matrix, Vector> matrix = TopicModelBase.loadModel(conf, modelPaths);
      Matrix modelMatrix = matrix.getFirst();
      modelMatrix = sparsifier.rebalance(modelMatrix);
      readModel = new SparseTopicModel(modelMatrix, eta, alpha, numUpdateThreads, modelWeight);
    } else {
      log.info("No model files found");
      throw new IOException("No model files found when computing perplexity!?!");
    }
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    MemoryUtil.stopMemoryLogger();
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
      throws IOException, InterruptedException{
    context.getCounter(CVB0Driver.Counters.SAMPLED_DOCUMENTS).increment(1);
    outKey.set(document.get().norm(1));
    DocTrainingState state = new DocTrainingState.Builder().setNumTopics(numTopics)
                                                           .setDocument(document.get())
                                                           .setMaxIters(maxIters).build();
    readModel.trainDocTopicModel(state);
    outValue.set(readModel.perplexity(state.getDocument(), state.getDocTopics()));
    context.write(outKey, outValue);
  }
}
