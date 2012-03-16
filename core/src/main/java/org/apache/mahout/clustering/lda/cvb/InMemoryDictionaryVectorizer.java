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

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.io.IOException;
import java.io.StringReader;
import java.util.List;

public class InMemoryDictionaryVectorizer implements DocumentVectorizer {
  private final Analyzer analyzer;
  private final OpenObjectIntHashMap<String> termIdMap;
  private final double[] termWeights;
  
  public InMemoryDictionaryVectorizer(Configuration conf, Path dictPath, Path weightPath, Analyzer analyzer) {
    SequenceFileDirIterable<Text, IntWritable> dictIter = new SequenceFileDirIterable<Text, IntWritable>(dictPath,
        PathType.GLOB, conf);
    List<Pair<Integer, String>> dictList = Lists.newArrayList();
    int max = -1;
    for(Pair<Text, IntWritable> pair : dictIter) {
      max = Math.max(max, pair.getSecond().get());
      dictList.add(Pair.of(pair.getSecond().get(), pair.getFirst().toString()));
    }
    String[] dict = new String[max + 1];
    for(Pair<Integer, String> pair : dictList) {
      dict[pair.getFirst()] = pair.getSecond();
    }
    termIdMap = mapify(dict);
    this.analyzer = analyzer;
    this.termWeights = null; // TODO: load from weightPath
  }
  
  private OpenObjectIntHashMap<String> mapify(String[] terms) {
    OpenObjectIntHashMap<String> termIdMap = new OpenObjectIntHashMap<String>(terms.length);
    for(int i=0; i<terms.length; i++) {
      termIdMap.put(terms[i], i);
    }
    return termIdMap;
  }
  
  public InMemoryDictionaryVectorizer(String[] terms, double[] weights, Analyzer analyzer) {
    termWeights = weights;
    termIdMap = mapify(terms);
    this.analyzer = analyzer;
  }
  
  @Override
  public void addToVector(String document, Vector outputVector) {
    TokenStream stream = analyzer.tokenStream(null, new StringReader(document));
    CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
    try {
      while (stream.incrementToken()) {
        if (termAtt.length() > 0) {
          String term = new String(termAtt.buffer(), 0, termAtt.length());
          if(termIdMap.containsKey(term)) {
            int termId = termIdMap.get(term);
            double value = termWeights != null ? termWeights[termId] : 1.0;
            outputVector.set(termId, outputVector.get(termId) + value);
          }
        }
      }
    } catch (IOException ioe) {
      // do something here?
    }
  }
}
