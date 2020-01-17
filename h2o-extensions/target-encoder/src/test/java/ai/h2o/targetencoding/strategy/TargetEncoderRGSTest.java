package ai.h2o.targetencoding.strategy;

import ai.h2o.targetencoding.TargetEncoderModel;
import hex.grid.Grid;
import hex.grid.GridSearch;
import hex.grid.HyperSpaceSearchCriteria;
import hex.grid.HyperSpaceWalker;
import hex.schemas.TargetEncoderV3;
import org.junit.BeforeClass;
import org.junit.Test;
import water.Job;
import water.Scope;
import water.TestUtil;
import water.api.GridSearchHandler;
import water.fvec.Frame;

import java.util.HashMap;

import static org.junit.Assert.*;

public class TargetEncoderRGSTest extends TestUtil {

  @BeforeClass
  public static void setup() {
    stall_till_cloudsize(1);
  }

  @Test
  public void getTargetEncodingMapByTrainingTEBuilder() {

    Scope.enter();
    try {
      HashMap<String, Object[]> hpGrid = new HashMap<>();
      hpGrid.put("blending", new Boolean[]{true, false});
      hpGrid.put("noise_level", new Double[]{0.0, 0.01,  0.1});
      hpGrid.put("k", new Double[]{1.0, 2.0, 3.0});
      hpGrid.put("f", new Double[]{5.0, 10.0, 20.0});
      
      TargetEncoderModel.TargetEncoderParameters parameters = new TargetEncoderModel.TargetEncoderParameters();

      GridSearchHandler.DefaultModelParametersBuilderFactory<TargetEncoderModel.TargetEncoderParameters, TargetEncoderV3.TargetEncoderParametersV3> modelParametersBuilderFactory = new GridSearchHandler.DefaultModelParametersBuilderFactory<>();

      HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria hyperSpaceSearchCriteria = new HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria();
      HyperSpaceWalker.RandomDiscreteValueWalker<TargetEncoderModel.TargetEncoderParameters> walker = new HyperSpaceWalker.RandomDiscreteValueWalker<>(parameters, hpGrid, modelParametersBuilderFactory, hyperSpaceSearchCriteria);

      HyperSpaceWalker.HyperSpaceIterator<TargetEncoderModel.TargetEncoderParameters> iterator = walker.iterator();
      int count = 0;
      while (iterator.hasNext(null)) {
        TargetEncoderModel.TargetEncoderParameters targetEncoderParameters = iterator.nextModelParameters(null);
        System.out.println( targetEncoderParameters._blending + ":" +  targetEncoderParameters._noise_level + ":" +  targetEncoderParameters._k + ":" +  targetEncoderParameters._f);
        count++;
      }
      assertEquals("Unexpected number of grid items", 54, count);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void regularGSOverTEParameters() {

    Scope.enter();
    try {
      Frame trainingFrame = parse_test_file("./smalldata/gbm_test/titanic.csv");
      Scope.track(trainingFrame);
      String responseColumn = "survived";
      asFactor(trainingFrame, responseColumn);

      HashMap<String, Object[]> hpGrid = new HashMap<>();
      hpGrid.put("blending", new Boolean[]{true, false});
      hpGrid.put("noise_level", new Double[]{0.0, 0.01,  0.1});
      hpGrid.put("k", new Double[]{1.0, 2.0, 3.0});
      hpGrid.put("f", new Double[]{5.0, 10.0, 20.0});

      TargetEncoderModel.TargetEncoderParameters parameters = new TargetEncoderModel.TargetEncoderParameters();
      parameters._train = trainingFrame._key;
      parameters._response_column = responseColumn;
      parameters._ignored_columns = ignoredColumns(trainingFrame, "home.dest", "embarked", parameters._response_column);

      GridSearchHandler.DefaultModelParametersBuilderFactory<TargetEncoderModel.TargetEncoderParameters, TargetEncoderV3.TargetEncoderParametersV3> modelParametersBuilderFactory = new GridSearchHandler.DefaultModelParametersBuilderFactory<>();

      HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria hyperSpaceSearchCriteria = new HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria();
      HyperSpaceWalker.RandomDiscreteValueWalker<TargetEncoderModel.TargetEncoderParameters> walker = new HyperSpaceWalker.RandomDiscreteValueWalker<>(parameters, hpGrid, modelParametersBuilderFactory, hyperSpaceSearchCriteria);

      Job<Grid> gs = GridSearch.startGridSearch(null, walker, 1);

      Scope.track_generic(gs);
      final Grid grid = gs.get();
      Scope.track_generic(grid);

      assertEquals(54, grid.getModelCount());

    } finally {
      Scope.exit();
    }
  }

  @Test // Probably could be resolved with redesign/fix PUBDEV-7110
  public void regularGSOverTEParameters_parallel() {

    Scope.enter();
    try {
      Frame trainingFrame = parse_test_file("./smalldata/gbm_test/titanic.csv");
      Scope.track(trainingFrame);
      String responseColumn = "survived";
      asFactor(trainingFrame, responseColumn);

      HashMap<String, Object[]> hpGrid = new HashMap<>();
      hpGrid.put("blending", new Boolean[]{true, false});
      hpGrid.put("noise_level", new Double[]{0.0, 0.01,  0.1});
      hpGrid.put("k", new Double[]{1.0, 2.0, 3.0});
      hpGrid.put("f", new Double[]{5.0, 10.0, 20.0});

      TargetEncoderModel.TargetEncoderParameters parameters = new TargetEncoderModel.TargetEncoderParameters();
      parameters._train = trainingFrame._key;
      parameters._response_column = responseColumn;
      parameters._ignored_columns = ignoredColumns(trainingFrame, "home.dest", "embarked", parameters._response_column);

      GridSearchHandler.DefaultModelParametersBuilderFactory<TargetEncoderModel.TargetEncoderParameters, TargetEncoderV3.TargetEncoderParametersV3> modelParametersBuilderFactory = new GridSearchHandler.DefaultModelParametersBuilderFactory<>();

      HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria hyperSpaceSearchCriteria = new HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria();
      HyperSpaceWalker.RandomDiscreteValueWalker<TargetEncoderModel.TargetEncoderParameters> walker = new HyperSpaceWalker.RandomDiscreteValueWalker<>(parameters, hpGrid, modelParametersBuilderFactory, hyperSpaceSearchCriteria);

      Job<Grid> gs = GridSearch.startGridSearch(null, walker, 2);

      Scope.track_generic(gs);
      final Grid grid = gs.get();
      Scope.track_generic(grid);

      assertEquals(54, grid.getModelCount());

    } finally {
      Scope.exit();
    }
  }

}
