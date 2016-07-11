package edu.uts;
/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/
public class leastSquaredGradient extends gradient{

    /************************************************************************************************
     * Function compute: computes least square gradient
     *          gradient = 1/2 * |label - data*parameter|^2
     *
     * @param data
     * @param label
     * @param parameter
     * @param weight
     *
     * @return double[]
     ***********************************************************************************************/
    @Override
    public double[] compute(double[] data, double label, double[] parameter, double weight) {
        double diff = dot(data, parameter) - label;
        double dLoss = diff * diff / 2.0;
        double[] dGradient = scal(diff, data);

        dGradient[data.length] = dLoss;

        return dGradient;
    }
}