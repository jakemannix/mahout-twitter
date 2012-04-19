package org.apache.mahout.clustering.lda.cvb;

import java.util.Iterator;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

public class SparseTopicModel extends TopicModelBase {
  
  private final int numTerms;

  protected SparseTopicModel(int numTopics, int numTerms) {
    super(numTopics);
    this.numTerms = numTerms;
  }

  @Override
  public void update(DocTrainingState state) {
    //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public Vector infer(DocTrainingState state, double minRelPerplexityDiff, int maxIters) {
    return null;  //To change body of implemented methods use File | Settings | File Templates.
  }

  /**
   * p(t|doc) : dense
   */
  @Override
  public void trainDocTopicModel(DocTrainingState state) {
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
        Vector docTopicModelRow = docTopicModel.viewRow(x);
        docTopicModelRow.setQuick(e.index(), docTopicModelRow.getQuick(e.index()) * e.get());
      }
    }
    // now recalculate p(topic|doc) by summing contributions from all of pTopicGivenTerm
    topics.assign(0.0);
    for(int x = 0; x < numTopics; x++) {
      it = original.iterateNonZero();
      double norm = 0;
      while(it.hasNext() && (e = it.next())!= null && e.index() < numTerms) {
        norm += docTopicModel.get(x, e.index());
      }
      topics.set(x, norm);
    }
    // now renormalize so that sum_x(p(x|doc)) = 1
    topics.assign(Functions.mult(1.0 / topics.norm(1)));
  }

  private void normalizeByTopic(Vector original, Matrix docTopicModel) {
  }

  private void pTopicGivenTerm(Vector original, Vector topics, Matrix docTopicModel) {
  }

  @Override
  public double perplexity(Vector document, Vector docTopics) {
    return 0;  //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public Matrix getTopicTermCounts() {
    return null;  //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public Vector expectedTermCounts(Vector original, Vector docTopics) {
    return null;  //To change body of implemented methods use File | Settings | File Templates.
  }
}
