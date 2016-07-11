package edu.uts;

import java.util.Iterator;
import java.util.Set;

/************************************************************************************************
 * Refactored by quan on 4/11/2016.
 ***********************************************************************************************/
public class CTF_SGDThread extends TF_SGDThread {
    protected sparseCoupledTensor spCTensor;
    protected factor[][] factors;
    protected int nTensorIdx;
    protected int nNumberOfTensor;
    protected long[] plDataSize;

    /************************************************************************************************
     * Function setDataSize: sets tensor Size
     *
     * @param tensorSize
     ***********************************************************************************************/
    public void setDataSize(long[] tensorSize){plDataSize = tensorSize;}

    /************************************************************************************************
     * Function CTF_SGDThread: initializes CTF_SGDThread class
     ***********************************************************************************************/
    public CTF_SGDThread(){
        spCTensor = null;
        factors = null;
        nTensorIdx = -1;
        nNumberOfTensor = 0;
        assert (nNumberOfTensor!=0);
    }

    /************************************************************************************************
     * Function CTF_SGDThread: initializes CTF_SGDThread class with the below parameters
     *
     * @param numberOfTensor
     * @param tensorIdx
     * @param modeIdx
     * @param cTensor
     * @param pfactors
     * @param rank
     * @param numberOfThread
     * @param threadIdx
     ***********************************************************************************************/
    public CTF_SGDThread(int numberOfTensor, int tensorIdx, int modeIdx, sparseCoupledTensor cTensor, factor[][] pfactors, int rank, int numberOfThread, int threadIdx) {
        this.nNumberOfTensor = numberOfTensor;
        this.spCTensor = cTensor;
        this.factors = pfactors;
        this.nTensorIdx = tensorIdx;
        nModeIdx = modeIdx;
        nRank = rank;
        nThreadIdx = threadIdx;
        nNumberOfThread = numberOfThread;

        int nLength;
        //Create localFactor data
        if(nThreadIdx<nNumberOfThread-1) {
            nLength = (int) Math.floor(1.0 * factors[nTensorIdx][nModeIdx].getLength() / nNumberOfThread);
            nStartIdx = nThreadIdx * nLength;
        }
        else//last part => get all the remaining
        {
            nLength = factors[nTensorIdx][nModeIdx].getLength() - nThreadIdx * (int) Math.floor(1.0 * factors[nTensorIdx][nModeIdx].getLength() / nNumberOfThread);;
            nStartIdx = nThreadIdx * (int) Math.floor(1.0 * factors[nTensorIdx][nModeIdx].getLength() / nNumberOfThread);
        }
        nEndIdx = nStartIdx + nLength;

        assert(nEndIdx<=factors[nTensorIdx][nModeIdx].getLength());
        localFactor = new factor(nLength, nRank);

        //Init localFactor
        for(int rowIdx=0; rowIdx<nLength; rowIdx++)
            localFactor.setRow(rowIdx, factors[nTensorIdx][nModeIdx].getRow(rowIdx + nStartIdx));

        //Reset dLoss =0
        dLoss = 0;
    }

    /************************************************************************************************
     * Function run
     ***********************************************************************************************/
    @Override
    public void run() {
        if(nTensorIdx == 0){
            if(nModeIdx<nNumberOfTensor-1){
                updateCoupledFactors();
            } // Coupled Mode
            else{
                updateFactors();
            }
        }
        else{
            updateFactors();
        }
    }

    /************************************************************************************************
     * Function updateFactors: updates factors
     *
     * return void
     ***********************************************************************************************/
    protected void updateFactors(){
        int nLength = nEndIdx - nStartIdx;

        for(int rowIdx=0; rowIdx<nLength; rowIdx++)
        {
            Set setObservedEntry = spCTensor.getObservedEntryIdx(nTensorIdx, nModeIdx, nStartIdx+rowIdx);
            int size = setObservedEntry.size();

            double[] label = new double[size];
            double[][] data = new double[size][];
            for(int j=0; j<size; j++)
                data[j] = new double[nRank];
            double[] parameter = localFactor.getRow(rowIdx);
            double[] weight = new double[size];

            int cnt = 0;
            Iterator<String> iterator = setObservedEntry.iterator();
            while(iterator.hasNext())
            {
                String strEntryIdx = iterator.next();
                String[] pEntryIdx = strEntryIdx.split(",");
                int[] index = new int[pEntryIdx.length];
                for (int j=0; j<pEntryIdx.length; j++)
                    index[j] = Integer.parseInt(pEntryIdx[j]);

                label[cnt] = spCTensor.get(nTensorIdx, index);

                for (int r=0; r<nRank; r++) {
                    double dotValue = 1;

                    for (int j=0; j<pEntryIdx.length; j++){
                        if (j!=nModeIdx)
                            dotValue*=factors[nTensorIdx][j].get(index[j], r);
                    }

                    data[cnt][r] = dotValue;
                }

                weight[cnt] = (double)1/plDataSize[nTensorIdx];

                cnt++;
            }

            //Update parameters
            localFactor.setRow(rowIdx, optimize(data, label, parameter, weight));
        }
    }

    /************************************************************************************************
     * Function updateCoupledFactors: updates coupled factors
     *
     * return void
     ***********************************************************************************************/
    protected void updateCoupledFactors(){
        assert (nTensorIdx==0&&nModeIdx+1<nNumberOfTensor);
        int nMainTensorIdx = nTensorIdx;
        int nMainTensorModeIdx = nModeIdx;
        int nCoupledTensorIdx = nModeIdx+1;
        int nCoupledTensorModeIdx = 0;

        int nLength = nEndIdx - nStartIdx;

        for(int rowIdx=0; rowIdx<nLength; rowIdx++)
        {
            Set setObservedEntry1 = spCTensor.getObservedEntryIdx(nMainTensorIdx, nMainTensorModeIdx, nStartIdx+rowIdx);
            int size1 = setObservedEntry1.size();

            Set setObservedEntry2 = spCTensor.getObservedEntryIdx(nCoupledTensorIdx, nCoupledTensorModeIdx, nStartIdx+rowIdx);
            int size2 = setObservedEntry2.size();
            int size = size1 + size2;

            double[] label = new double[size];
            double[][] data = new double[size][];
            for(int j=0; j<size; j++)
                data[j] = new double[nRank];
            double[] parameter = localFactor.getRow(rowIdx);
            double[] weight = new double[size];

            int cnt = 0;
            //Main Tensor
            Iterator<String> iterator1 = setObservedEntry1.iterator();
            while(iterator1.hasNext())
            {
                String strEntryIdx = iterator1.next();
                String[] pEntryIdx = strEntryIdx.split(",");
                int[] index = new int[pEntryIdx.length];
                for (int j=0; j<pEntryIdx.length; j++)
                    index[j] = Integer.parseInt(pEntryIdx[j]);

                label[cnt] = spCTensor.get(nMainTensorIdx, index);

                for (int r=0; r<nRank; r++) {
                    double dotValue = 1;

                    for (int j=0; j<pEntryIdx.length; j++){
                        if (j!=nMainTensorModeIdx)
                            dotValue*=factors[nMainTensorIdx][j].get(index[j], r);
                    }

                    data[cnt][r] = dotValue;
                }
                weight[cnt] = (double)1/plDataSize[nMainTensorIdx];

                cnt++;
            }

            //Coupled Tensor
            Iterator<String> iterator2 = setObservedEntry2.iterator();
            while(iterator2.hasNext())
            {
                String strEntryIdx = iterator2.next();
                String[] pEntryIdx = strEntryIdx.split(",");
                int[] index = new int[pEntryIdx.length];
                for (int j=0; j<pEntryIdx.length; j++)
                    index[j] = Integer.parseInt(pEntryIdx[j]);

                label[cnt] = spCTensor.get(nCoupledTensorIdx, index);

                for (int r=0; r<nRank; r++) {
                    double dotValue = 1;

                    for (int j=0; j<pEntryIdx.length; j++){
                        if (j!=nCoupledTensorModeIdx)
                            dotValue*=factors[nCoupledTensorIdx][j].get(index[j], r);
                    }

                    data[cnt][r] = dotValue;
                }
                weight[cnt] = (double)1/plDataSize[nCoupledTensorIdx];

                cnt++;
            }

            //Update parameters
            localFactor.setRow(rowIdx, optimize(data, label, parameter, weight));
        }
    }
}
