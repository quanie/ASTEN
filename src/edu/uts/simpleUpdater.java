package edu.uts;

import java.util.Arrays;

/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/
public class simpleUpdater extends updater{
    /************************************************************************************************
     * Function compute: computes updater
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
        updatedParameter = Arrays.copyOf(oldParameter, oldParameter.length);

        for (int i=0; i<oldParameter.length; i++){
            updatedParameter[i] -= stepSize * gradient[i];
        }
        return 0;
    }
}
