package edu.uts;

/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/
public abstract class gradient{
    //the last entry is loss
    public abstract double[] compute(double[] data, double label, double []parameter, double weight);

    /************************************************************************************************
     * Function compute: computes gradient
     *
     * @param data
     * @param label
     * @param parameter
     *
     * @return double[]
     ***********************************************************************************************/
    public double[] compute(double[] data, double label, double []parameter){return compute(data, label, parameter, 1);}

    /************************************************************************************************
     * Function dot: computes dot product of two vectors
     *
     * @param x
     * @param y
     *
     * @return double
     ***********************************************************************************************/
    public double dot(double[] x, double[] y){
        assert (x.length == y.length);
        double dResult = 0;

        for(int i=0; i<x.length; i++)
            dResult += x[i] * y[i];

        return dResult;
    }

    /************************************************************************************************
     * Function scal: computes vector scaling
     *
     * @param a
     * @param y
     *
     * @return double[]
     ***********************************************************************************************/
    public double[] scal(double a, double[] y){
        double[] dResult = new double[y.length + 1];

        for(int i=0; i<y.length; i++)
            dResult[i] = a * y[i];

        return dResult;
    }
}