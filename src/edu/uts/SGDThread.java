package edu.uts;

/************************************************************************************************
 * Refactored by quan on 4/8/2016.
 ***********************************************************************************************/
public abstract class SGDThread extends gradientDescent implements Runnable {
    //Local variables
    protected factor localFactor;
    protected int nStartIdx;        //start Index of localFactor
    protected int nEndIdx;          //end Index of localFactor

    public int getStartIdx(){return nStartIdx;}
    public int getEndIdx(){return nEndIdx;}
    public factor getLocalFactor(){return localFactor;}

    public abstract void run();
}
