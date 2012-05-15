package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Preconditions;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;

public class DocTrainingState {

  private double perplexity;

  private final Vector document;
  
  private final SparseMatrix docTopicModel;
  
  private final Vector docTopics;

  private final int maxIters;

  public int getMaxIters() {
    return maxIters;
  }
  
  private DocTrainingState(Vector document, Vector docTopics, SparseMatrix docTopicModel, int maxIters) {
    this.document = document;
    this.docTopics = docTopics;
    this.docTopicModel = docTopicModel;
    this.maxIters = maxIters;
  }

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

  public SparseMatrix getDocTopicModel() {
    return docTopicModel;
  }

  public Vector getDocTopics() {
    return docTopics;
  }

  public static class Builder {
    private Vector document;
    private SparseMatrix docTopicModel;
    private Vector docTopics;
    private int numTopics = -1;
    private int maxIters = 0;

    public Builder setDocument(Vector document) {
      this.document = document;
      return this;
    }

    public Builder setDocTopicModel(SparseMatrix docTopicModel) {
      this.docTopicModel = docTopicModel;
      return this;
    }

    public Builder setDocTopics(Vector docTopics) {
      this.docTopics = docTopics;
      return this;
    }

    public Builder setMaxIters(int maxIters) {
      this.maxIters = maxIters;
      return this;
    }
    
    public Builder setNumTopics(int numTopics) {
      this.numTopics = numTopics;
      return this;
    }

    public DocTrainingState build() {
      Preconditions.checkNotNull(document, "Document must not be null");
      if(docTopics == null) {
        Preconditions.checkState(numTopics > 0, "numTopics must be > 0");
        docTopics = new DenseVector(numTopics).assign(1.0 / numTopics);
      } else {
        Preconditions.checkState(numTopics < 0 || numTopics == docTopics.size(),
                                 "cannot set numTopics and docTopics");
        numTopics = docTopics.size();
      }
      if(docTopicModel == null) {
        docTopicModel = new SparseMatrix(document.size(), numTopics);
      }
      return new DocTrainingState(document, docTopics, docTopicModel, maxIters);
    }
  }
}
