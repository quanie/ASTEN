package edu.uts;

import java.util.Arrays;

import static java.lang.Math.signum;
/************************************************************************************************
 * Refactored by quan on 4/11/2016.
 ***********************************************************************************************/

/************************************************************************************************
 * Updater for L1 regularized problems.
 *          R(w) = ||w||_1
 * Uses a step-size decreasing with the square root of the number of iterations.

 * Instead of subgradient of the regularizer, the proximal operator for the
 * L1 regularization is applied after the gradient step. This is known to
 * result in better sparsity of the intermediate solution.
 *
 * The corresponding proximal operator for the L1 norm is the soft-thresholding
 * function. That is, each weight component is shrunk towards 0 by shrinkageVal.
 *
 * If w >  shrinkageVal, set weight component to w-shrinkageVal.
 * If w < -shrinkageVal, set weight component to w+shrinkageVal.
 * If -shrinkageVal < w < shrinkageVal, set weight component to 0.
 *
 * Equivalently, set weight component to signum(w) * max(0.0, abs(w) - shrinkageVal)
 ***********************************************************************************************/
public class L1Updater extends updater {

    /************************************************************************************************
     * Function compute: computes L1 update
     *
     * @param oldParameter - Column matrix of size dx1 where d is the number of features.
     * @param gradient - Column matrix of size dx1 where d is the number of features.
     * @param stepSize - step size across iterations
     * @param regParam - Regularization parameter
     *
     * @return double
     ***********************************************************************************************/
    @Override
    public double compute(double[] oldParameter, double[] gradient, double stepSize, double regParam) {
        double shrinkageVal = regParam * stepSize;
        double L1Norm = 0;

        updatedParameter = Arrays.copyOf(oldParameter, oldParameter.length);

        for (int i=0; i<oldParameter.length; i++){
            // Take gradient step
            updatedParameter[i] -= stepSize * gradient[i];

            // Apply proximal operator (soft thresholding)
            double wi = updatedParameter[i];
            updatedParameter[i] = signum(wi) * Math.max(0.0, Math.abs(wi) - shrinkageVal);

            L1Norm += Math.abs(updatedParameter[i]);
        }

        return L1Norm * regParam;
    }
}
