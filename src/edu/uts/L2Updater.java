package edu.uts;

import java.util.Arrays;
/************************************************************************************************
 * Refactored by quan on 4/11/2016.
 ***********************************************************************************************/
/************************************************************************************************
 * Updater for L2 regularized problems.
 *          R(w) = 1/2 ||w||^2
 * Uses a step-size decreasing with the square root of the number of iterations.
 ***********************************************************************************************/
public class L2Updater extends updater {
    /************************************************************************************************
     * Function compute: computes L2 update
     * @param oldParameter - Column matrix of size dx1 where d is the number of features.
     * @param gradient - Column matrix of size dx1 where d is the number of features.
     * @param stepSize - step size across iterations
     * @param regParam - Regularization parameter
     *
     * @return double
     ***********************************************************************************************/
    @Override
    public double compute(double[] oldParameter, double[] gradient, double stepSize, double regParam) {
        double L2Norm = 0;
        updatedParameter = Arrays.copyOf(oldParameter, oldParameter.length);

        for (int i=0; i<oldParameter.length; i++){
            // add up both updates from the gradient of the loss (= step) as well as
            // the gradient of the regularizer (= regParam * weightsOld)
            // w' = w - thisIterStepSize * (gradient + regParam * w)
            // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
            updatedParameter[i] *= (1.0 - stepSize * regParam);
            updatedParameter[i] -= stepSize * gradient[i];

            L2Norm += updatedParameter[i]*updatedParameter[i];
        }

        return 0.5 * L2Norm * regParam;
    }
}
