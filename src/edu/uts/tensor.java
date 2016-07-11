package edu.uts;

/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/
public abstract class tensor {
    protected static final boolean CODING = true;     //debug at coding stage
    protected static final boolean DEBUG = false;     //debug at coding stage

    protected int nMode;
    protected int[] pnLength;

    /************************************************************************************************
     * Function getLength: gets length of the tensor at mode index
     *
     * @param modeIdx
     *
     * @return int
     ***********************************************************************************************/
    public int getLength(int modeIdx){
        if(CODING)
            assert(modeIdx < nMode);
        return pnLength[modeIdx];
    }
    public int getMode(){return nMode;}

    /********************************************************************************************
     * Function set: sets the value of the tensor at index
     *
     * @param index index
     * @param value value to be set
     *
     * Return:  int
     *********************************************************************************************/
    public abstract void set(int[] index, double value);

    /********************************************************************************************
     * Function set: sets value of an entry
     * @param i
     * @param j
     * @param value
     *******************************************************************************************/
    public void set(int i, int j, double value){
        assert (nMode==2);
        int[] index={i,j};
        set(index, value);
    }

    /********************************************************************************************
     * Function set: sets value of an entry
     * @param i
     * @param j
     * @param k
     * @param value
     *******************************************************************************************/
    public void set(int i, int j, int k, double value){
        assert (nMode==3);
        int[] index={i,j,k};
        set(index, value);
    }

    /********************************************************************************************
     * Function get: returns value of the tensor at index
     *
     * @param index index
     *
     * Return:  int
     *********************************************************************************************/
    public abstract double get(int[] index);

    /********************************************************************************************
     * Function get: returns value of the tensor at index
     *
     * @param i
     * @param j
     *
     * @return double
     *******************************************************************************************/
    public double get(int i, int j){
        assert (nMode==2);
        int[] index={i,j};
        return get(index);
    }

    /********************************************************************************************
     * Function get: returns value of the tensor at index
     *
     * @param i
     * @param j
     * @param k
     *
     * @return double
     *******************************************************************************************/
    public double get(int i, int j, int k){
        assert (nMode==3);
        int[] index={i,j,k};
        return get(index);
    }
}
