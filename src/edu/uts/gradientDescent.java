package edu.uts;

/************************************************************************************************
 * Refactored by quan on 4/8/2016.
 ***********************************************************************************************/
public abstract class gradientDescent {
    protected double dStepSize = 1.0;
    protected int nNumIteration = 1;
    protected double dRegParam = 0.0;

    protected gradient usedGradient;
    protected updater usedUpdater;
    protected double dLoss;

    protected abstract double[] optimize(double[][] data, double[] label, double[] parameter, double[] weight);

    public double getLoss(){return dLoss;}
    public void setStepSize(double step){dStepSize = step;}
    public void setNumIteration(int iter){nNumIteration = iter;}
    public void setRegParam(double regParam){dRegParam = regParam;}
    public void setGradient(gradient usedGradient){this.usedGradient = usedGradient;}
    public void setUpdater(updater usedUpdater){this.usedUpdater = usedUpdater;}
}
