package edu.uts;

import java.util.Arrays;
import java.util.Random;

import static java.lang.System.exit;

/************************************************************************************************
 * Refactored by quan on 4/8/2016.
 ***********************************************************************************************/
public class factor extends tensor {
    protected final int nLength;
    protected final int nRank;
    public int getRank(){return nRank;}
    public int getLength(){return nLength;}
    double[][] data;

    /************************************************************************************************
     * Function factor: initializes factor class with the following parameter
     *
     * @param length
     * @param rank
     *********************************************************************************************/
    factor(int length, int rank){
        nMode = 2;
        nLength = length;
        nRank = rank;

        //Init tensor's class default properties
        pnLength = new int[nMode];
        pnLength[0] = length;
        pnLength[1] = rank;

        data = new double[nLength][];
        for(int i=0; i<nLength; i++)
            data[i] = new double[nRank];
    }

    /************************************************************************************************
     * Function set: sets value of an entry of the factor
     *
     * @param rowIdx
     * @param rankIdx
     * @param value
     *********************************************************************************************/
    public void set(int rowIdx, int rankIdx, double value){
        data[rowIdx][rankIdx] = value;
    }

    /************************************************************************************************
     * Function setRow: sets a row of the factor
     *
     * @param rowIdx
     * @param value
     *********************************************************************************************/
    public void setRow(int rowIdx, double[] value){
        assert(rowIdx<nLength);
        assert(value.length==nRank);
        for(int r=0; r<nRank; r++)
            data[rowIdx][r] = value[r];
        //data[rowIdx] = Arrays.copyOf(value, nRank);
    }

    /************************************************************************************************
     * Function get: gets an entry's value
     *
     * @param rowIdx
     * @param rankIdx
     *
     * @return double
     *********************************************************************************************/
    public double get(int rowIdx, int rankIdx){
        return data[rowIdx][rankIdx];
    }

    /************************************************************************************************
     * Function getRow: gets a row of the factor
     *
     * @param rowIdx
     *
     * @return double[]
     *********************************************************************************************/
    public double[] getRow(int rowIdx){
        assert(rowIdx<nLength);
        return Arrays.copyOf(data[rowIdx], nRank);
    }

    /************************************************************************************************
     * Function set: sets an entry's value
     *
     * @param index index
     * @param value value to be set
     *
     *********************************************************************************************/
    @Override
    public void set(int[] index, double value) {
        assert(index.length==nMode);
        set(index[0], index[1], value);
    }

    /************************************************************************************************
     * Function get: gets value of an entry
     *
     * @param index index
     *
     * @return double
     *********************************************************************************************/
    @Override
    public double get(int[] index) {
        assert(index.length==nMode);
        return get(index[0], index[1]);
    }
    /********************************************************************************************
     * Function init: initializes tensor with value
     *
     * @param value
     *
     * Return:  void
     *********************************************************************************************/
    public void init(double value)
    {
        for(int i = 0; i < nLength; i++)
            for(int r=0; r<nRank; r++)
                data[i][r] = value;
    }
    /********************************************************************************************
     * Function reset: initializes tensor with mean 0
     *
     * Return:  void
     *********************************************************************************************/
    public void reset()
    {
        reset(0);
    }
    /********************************************************************************************
     * Function reset: initializes tensor with inputted mean
     *
     * @param mean reset with mean
     *
     * Return:  void
     *********************************************************************************************/
    public void reset(double mean)
    {
        int i,r;

        Random rand = new Random();
        for(i = 0; i < nLength; i++)
            for(r=0; r<nRank; r++){
                data[i][r] = (rand.nextGaussian()/2.0f + mean);

                if(CODING){
                    if(Double.isNaN(data[i][r])){
                        System.out.println("NaN Error on Reset");
                        exit(0);
                    }
                }
            }
    }
}
