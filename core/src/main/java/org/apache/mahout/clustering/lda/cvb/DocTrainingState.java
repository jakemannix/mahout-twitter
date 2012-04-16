package org.apache.mahout.clustering.lda.cvb;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public class DocTrainingState {

  private double perplexity;

  private Vector document;
  
  private Matrix docTopicModel;
  
  private Vector docTopics;

  public double getPerplexity() {
    return perplexity;
  }

  public DocTrainingState setPerplexity(double perplexity) {
    this.perplexity = perplexity;
    return this;
  }

  public Vector getDocument() {
    return document;
  }

  public DocTrainingState setDocument(Vector document) {
    this.document = document;
    return this;
  }

  public Matrix getDocTopicModel() {
    return docTopicModel;
  }

  public DocTrainingState setDocTopicModel(Matrix docTopicModel) {
    this.docTopicModel = docTopicModel;
    return this;
  }

  public Vector getDocTopics() {
    return docTopics;
  }

  public DocTrainingState setDocTopics(Vector docTopics) {
    this.docTopics = docTopics;
    return this;
  }
}
