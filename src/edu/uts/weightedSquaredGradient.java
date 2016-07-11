package edu.uts;

/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/
public class weightedSquaredGradient extends gradient{
    /************************************************************************************************
     * Function compute: computes weighted square gradient
     *        gradient = 1/2 * weight * |label - data*parameter|^2
     * @param data
     * @param label
     * @param parameter
     * @param weight
     * @return
     ***********************************************************************************************/
    @Override
    public double[] compute(double[] data, double label, double[] parameter, double weight) {
        double diff = dot(data, parameter) - label;
        double dLoss = weight*diff * diff / 2.0;
        double[] dGradient = scal(weight*diff, data);

        dGradient[data.length] = dLoss;

        return dGradient;
    }
}
