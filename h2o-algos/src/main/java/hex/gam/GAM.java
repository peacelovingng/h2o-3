package hex.gam;

import hex.ModelBuilder;
import hex.ModelCategory;
import hex.glm.GLMModel.GLMParameters.Family;
import hex.glm.GLMModel.GLMParameters.Link;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() { return new ModelCategory[]{ModelCategory.Regression}; }

  @Override
  public boolean isSupervised() { return true; }

  @Override
  public BuilderVisibility builderVisibility() { return BuilderVisibility.Experimental; }

  // Called from an http request

  
  @Override public boolean havePojo() { return false; }
  @Override public boolean haveMojo() { return false; }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }
  public GAM(GAMModel.GAMParameters parms) { super(parms);init(false); }
  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) { super(parms, key); init(false); }

  @Override public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      if (!_parms._family.equals(Family.gaussian)) 
        error("_family", "Only gaussian family is supported for now.");
      if (!_parms._link.equals(Link.identity) && !_parms._link.equals(Link.family_default))
        error("_link", "Only identity or family_default link is supported for now.");
    }
  }

  @Override
  public void checkDistributions() {  // will be called in ModelBuilder.java
    if (!_response.isNumeric()) {
      error("_response", "Expected a numerical response, but instead got response with " + _response.cardinality() + " categories.");
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return false; // no use, we don't output probabilities
  }

  @Override
  protected int init_getNClass() {
    return 1; // only regression is supported for now
  }

  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }
  
  @Override 
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  @Override protected void checkMemoryFootPrint_impl() {
    ;
  }
  private class GAMDriver extends Driver {
    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_x, expand it out to calculate the f(x) and attach to the frame.
     * @return
     */
    Frame adaptTrain() {
      Frame orig = _parms.train();  // contain all needed columns
      Frame adapt = new Frame(_parms.train());  // only contains predictors and response column
      
      return adapt;
    }
    
    @Override
    public void computeImpl() {
      init(true); //this can change the seed if it was set to -1
      // Something goes wrong
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      _job.update(0, "Initializing model training");
      
      buildModel(); // build gam model 
    }

    public final void buildModel() {
      GAMModel model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this));
      model.delete_and_lock(_job);

     Frame newTFrame = adaptTrain();  // get frames with correct predictors and spline functions
    }
  }
}
