package org.apache.mahout.math.function;

import org.apache.mahout.math.Vector;

public final class VectorFunctions {
  
  public static class Norm implements VectorFunction {
    private final double value;
    
    public Norm(double value) {
      this.value = value;
    }
    @Override
    public double apply(Vector f) {
      return f.norm(value);
    }
  }
  
  public static final VectorFunction NORM_1 = new Norm(1);
  
}
