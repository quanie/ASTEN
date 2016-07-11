package edu.uts;

/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/
public abstract class updater {
    protected double[] updatedParameter;
    public double[] getUpdatedParameter(){return updatedParameter;}

    /********************************************************************************************
     * Name: compute
     * Function: Compute an updated value for weights given the gradient, stepSize, iteration number and
     * regularization parameter. Also returns the regularization value regParam * R(w)
     * computed using the *updated* weights.
     *
     * @param oldParameter - Column matrix of size dx1 where d is the number of features.
     * @param gradient - Column matrix of size dx1 where d is the number of features.
     * @param stepSize - step size across iterations
     * @param regParam - Regularization parameter
     *
     * Return:  double regParam
     *********************************************************************************************/
    public abstract double compute(double[] oldParameter, double[] gradient, double stepSize, double regParam);
}
