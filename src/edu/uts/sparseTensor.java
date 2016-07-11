package edu.uts;
import java.util.*;
/************************************************************************************************
 * Refactored by quan on 4/11/2016.
 ***********************************************************************************************/

public class sparseTensor extends tensor{
    private int nObservedEntry; //number of observed entry
    protected HashMap<String, Double> data;
    protected Set[][] pObservedIdx; //indexes of observed data in each dimension

    /********************************************************************************************
     * Function sparseTensor: initializes sparseTensor instance
     *
     * @param mode mode of each tensor
     * @param length length of each mode
     *
     * Return:  void
     *********************************************************************************************/
    sparseTensor(int mode, int []length)
    {
        data = new HashMap<String, Double>(); //only store observed nonZero data
        nObservedEntry = 0;
        nMode = mode;
        pnLength = new int[nMode];
        pObservedIdx = new Set[nMode][];

        for (int i=0;i<nMode; i++)
        {
            pObservedIdx[i] = new Set[length[i]];
            pnLength[i] = length[i];

            for (int k=0; k<length[i];k++)
                pObservedIdx[i][k] = new HashSet();
        }
    }

    /********************************************************************************************
     * Function set: sets value of the tensorIdx at index[]
     *
     * @param index index
     * @param value value to be set
     *
     * Return:  void
     *********************************************************************************************/
    @Override
    public void set(int[] index, double value) {
        assert(index.length==nMode);

        String key = getKey(index);

        if(value!=0)
        {
            data.put(key, value);
        }

        for (int modeIdx=0; modeIdx<index.length; modeIdx++)
        {
            //Add to pObservedIdx
            String NNZIdxKey = getObservedIdxKey(index, modeIdx);
            pObservedIdx[modeIdx][index[modeIdx]].add(NNZIdxKey);
        }
        nObservedEntry++;
    }

    /********************************************************************************************
     * Function get: returns value at index[]
     *
     * Return:  double
     *********************************************************************************************/
    @Override
    public double get(int[] index)
    {
        assert(index.length==nMode);

        String key = getKey(index);

        if (data.containsKey(key))
            return data.get(key);
        else
            return 0;
    }

    /********************************************************************************************
     * Function getKey: calculates the key
     *
     * @param index index
     *
     * Return:  String
     *********************************************************************************************/
    protected String getKey(int []index)
    {
        String sKey ="";
        for(int i=0; i<index.length; i++)
        {
            if(i==0)
                sKey = String.valueOf(index[i]);
            else
                sKey += "," + index[i];
        }
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
     * Function getObservedIdx: returns the observed Indexes
     *
     * Return:  int
     *********************************************************************************************/
    public Set getObservedIdx(int modeIdx, int rowIdx)
    {
        return pObservedIdx[modeIdx][rowIdx];
    }

    /********************************************************************************************
     * Function getObservedEntryNumber: gets number of observed entry
     *
     * Return:  int
     *********************************************************************************************/
    public int getObservedEntryNumber()
    {
        return nObservedEntry;
    }
    /********************************************************************************************
     * Function checkObservedEntryIdx: checks if the Idx of Observed Entry of sparse tensor is stored correctly?
     *
     * Return:  boolean
     *********************************************************************************************/
    public boolean checkObservedEntryIdx()
    {
        boolean bIsOK =true;
        int[] pObservedEntrySize = new int[nMode];

        for(int modeIdx=0;modeIdx<nMode;modeIdx++)
        {
            pObservedEntrySize[modeIdx] = 0;

            for(int rowIdx=0; rowIdx<pnLength[modeIdx]; rowIdx++)
                pObservedEntrySize[modeIdx]+= pObservedIdx[modeIdx][rowIdx].size();

            if(!(pObservedEntrySize[modeIdx]==nObservedEntry))
                bIsOK = false;
        }

        return bIsOK;
    }
}

