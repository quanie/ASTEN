package edu.uts;

import java.util.Set;
/************************************************************************************************
 * Refactored by quan on 4/13/2016.
 ***********************************************************************************************/

/********************************************************************************************
 * Name: coupledTensor
 * Function: container of all coupled tensors
 * Note: Tensor 0 is the main tensor, other tensors (tensor 1, 2, 3, ...) are coupled with
 *           the main tensor as the following rule:
 *           - Mode 0 of tensor 0 is coupled with mode 0 of tensor 1
 *           - Mode 1 of tensor 0 is coupled with mode 0 of tensor 2
 *           - Mode 2 of tensor 0 is coupled with mode 0 of tensor 3
 *           - Mode 3 of tensor 0 is coupled with mode 0 of tensor 4
 *           - ....
 *           => Mode i of tensor 0 is coupled with mode 0 of tensor i+1
 *               & mode 0 of tensor i is coupled with mode i-1 of tensor 0
 *********************************************************************************************/
public abstract class coupledTensor {
    protected static final boolean CODING = true;     //debug at coding stage
    protected static final boolean DEBUG = false;     //debug at coding stage

    protected int nNumberOfTensor;
    protected int[] pnMode; //all mode of the coupled tensors
    protected int[][] pnLength;//all length of all modes of all tensors
    protected long[] plObservedEntry; //number of observed entry of each Tensor

    protected Set[][][] pObservedIdx; //indexes of observed data in each dimension of each tensor
    /********************************************************************************************
     * Function get: return value of the tensorIdx at index[]
     *
     * @param tensorIdx index of the tensor to be extracted its value
     * @param index index
     *
     * Return:  double
     *********************************************************************************************/
    public abstract double get(int tensorIdx, int[] index);
    /********************************************************************************************
     * Function set: set value of the tensorIdx at index[]
     * @param tensorIdx index of the tensor to be extracted its value
     * @param index index
     * @param value value to be set at []index
     *
     * Return:  void
     *********************************************************************************************/
    public abstract void set(int tensorIdx, int[] index, double value);
    /********************************************************************************************
     * Function getNumberOfTensor: return the number of tensor
     *
     * Return:  int
     *********************************************************************************************/
    public int getNumberOfTensor(){
        return nNumberOfTensor;
    }
    /********************************************************************************************
     * Function getMode: return the mode of the tensorIdx
     *
     * @param tensorIdx
     *
     * Return:  int
     *********************************************************************************************/
    public int getMode(int tensorIdx){
        if (CODING)
            assert(tensorIdx < nNumberOfTensor);

        return pnMode[tensorIdx];
    }

    /********************************************************************************************
     * Function getLength: return the length of modeIdx of the tensorIdx
     *
     * @param tensorIdx
     * @param modeIdx
     *
     * Return:  int
     *********************************************************************************************/
    public int getLength(int tensorIdx, int modeIdx){
        if (CODING)
            assert(tensorIdx < nNumberOfTensor);

        return pnLength[tensorIdx][modeIdx];
    }
    /********************************************************************************************
     * Function getObservedEntryNumber: get number of observed Entry of a tensor
     *
     * @param tensorIdx index of the tensor to be extracted the number of observed Entry
     *
     * Return:  int
     *********************************************************************************************/
    public long getObservedEntryNumber(int tensorIdx) {return plObservedEntry[tensorIdx];}
    public long[] getObservedEntryNumber() {return plObservedEntry;}
    /********************************************************************************************
     * Function getObservedEntryIdx: get indexes of all Observed Entries of a mode, at a row of a tensor
     *
     * @param tensorIdx index of the tensor
     * @param modeIdx index of the mode
     * @param rowIdx index of the row
     *
     * Return:  Set
     *********************************************************************************************/
    public Set getObservedEntryIdx(int tensorIdx, int modeIdx, int rowIdx) {return pObservedIdx[tensorIdx][modeIdx][rowIdx];}
}
