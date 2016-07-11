package edu.uts;
import java.util.*;
/************************************************************************************************
 * Refactored by quan on 4/11/2016.
 ***********************************************************************************************/

public class sparseCoupledTensor extends coupledTensor{
    private HashMap<String, Double> nonZeroData;
    /********************************************************************************************
     * Function sparseCoupledTensor: initializes sparseCoupledTensor instance
     *
     * @param numOfTensor: number of tensor
     * @param mode: mode of each tensor
     * @param length length of each mode
     *
     * Return:  void
     *********************************************************************************************/
    sparseCoupledTensor(int numOfTensor, int[] mode, int[][] length)
    {
        nNumberOfTensor = numOfTensor;
        pnMode = new int[nNumberOfTensor];
        pnLength = new int[nNumberOfTensor][];
        plObservedEntry = new long[nNumberOfTensor];
        pObservedIdx = new Set[nNumberOfTensor][][];

        for (int i=0; i<nNumberOfTensor; i++)
        {
            pnMode[i] = mode[i];
            plObservedEntry[i]=0;
            pnLength[i] = new int[pnMode[i]];
            pObservedIdx[i] = new Set[pnMode[i]][];
            for (int j=0; j<pnMode[i]; j++)
            {
                pnLength[i][j] = length[i][j];
                pObservedIdx[i][j] = new Set[pnLength[i][j]];
                for (int k=0; k<pnLength[i][j];k++)
                    pObservedIdx[i][j][k] = new HashSet();
            }
        }

        //check if data is formatted correctly
        for(int i=1; i<nNumberOfTensor; i++)
            assert(pnLength[i][0]==pnLength[0][i-1]);

        nonZeroData = new HashMap<>();
    }
    /********************************************************************************************
     * Function get: returns value of the tensorIdx at index[]
     *
     * @param tensorIdx index of the tensor to be extracted its value
     * @param index index
     *
     * Return:  double
     *********************************************************************************************/
    @Override
    public double get(int tensorIdx, int[] index)
    {
        double dReturn = 0;
        assert(tensorIdx < nNumberOfTensor);
        assert(index.length==pnMode[tensorIdx]);

        String key = getKey(tensorIdx, index);

        if (nonZeroData.containsKey(key))
            dReturn = nonZeroData.get(key);
        else{
            //error?
        }
        return dReturn;
    }

    /********************************************************************************************
     * Function set: sets value of the tensorIdx at index[]
     *
     * @param tensorIdx index of the tensor to be extracted its value
     * @param index index
     * @param value value to be set at []index
     *
     * Return:  void
     *********************************************************************************************/
    @Override
    public void set(int tensorIdx, int[] index, double value) {
        assert(tensorIdx < nNumberOfTensor);
        assert(index.length==pnMode[tensorIdx]);

        String key = getKey(tensorIdx, index);

        if(value!=0)
            nonZeroData.put(key, value);

        for (int i=0; i<index.length; i++) {
            //Add to pObservedIdx
            String observedIdxKey = getObservedIdxKey(index, i);
            pObservedIdx[tensorIdx][i][index[i]].add(observedIdxKey);
        }

        plObservedEntry[tensorIdx]++;
    }

    /********************************************************************************************
     * Function getKey: calculates the key
     *
     * Return:  String
     *********************************************************************************************/
    protected String getKey(int tensorIdx, int []index)
    {
        String sKey = String.valueOf(tensorIdx);

        for(int i=0; i<index.length; i++)
            sKey += "," + index[i];

        return sKey;
    }

    /********************************************************************************************
     * Function getObservedIdxKey: calculates the key for observedIdx
     *
     * Return:  String
     *********************************************************************************************/
    protected String getObservedIdxKey(int []index, int nDimension)
    {
        String sKey="";
        boolean bIsAdd = false;

        for(int i=0; i<index.length; i++)
        {
            if(bIsAdd)
                sKey += "," + index[i];
            else
            {
                sKey += index[i];
                bIsAdd = true;
            }
        }

        return sKey;
    }

    /********************************************************************************************
     * Function checkObservedEntryIdx: checks if the Idx of Observed Entry of coupled tensor is stored correctly?
     *
     * Return:  boolean
     *********************************************************************************************/
    public boolean checkObservedEntryIdx()
    {
        boolean bIsOK =true;
        int i,j,k;
        int[][] pObservedEntrySize;

        pObservedEntrySize = new int[nNumberOfTensor][];
        for (i=0; i<nNumberOfTensor; i++)
        {
            pObservedEntrySize[i] = new int[pnMode[i]];
            for(j=0;j<pnMode[i];j++)
            {
                pObservedEntrySize[i][j] = 0;

                for(k=0; k<pnLength[i][j]; k++)
                    pObservedEntrySize[i][j]+= pObservedIdx[i][j][k].size();

                if(!(pObservedEntrySize[i][j]==plObservedEntry[i]))
                    bIsOK = false;
            }
        }
        return bIsOK;
    }
}
