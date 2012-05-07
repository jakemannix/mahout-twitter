package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

public class SparsifyingVectorSumReducer
    extends Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {
  public static final String SPARSIFIER_CLASS = "sparsifierClass";
  private VectorSparsifier sparsifier;

  @Override
  public void setup(Context ctx) {
    Class<? extends VectorSparsifier> sparsifierClass = BackgroundFrequencyVectorSparsifier.class;
/*        ctx.getConfiguration().getClass(SPARSIFIER_CLASS,
                                        NoopVectorSparsifier.class,
                                        VectorSparsifier.class); */

    sparsifier = ReflectionUtils.newInstance(sparsifierClass, ctx.getConfiguration());
  }

  @Override
  protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context ctx)
      throws IOException, InterruptedException {
    Vector vector = null;
    for (VectorWritable v : values) {
      if (vector == null) {
        vector = v.get();
      } else {
        vector.assign(v.get(), Functions.PLUS);
      }
    }
    ctx.write(key, new VectorWritable(sparsifier.sparsify(vector)));
  }

}
