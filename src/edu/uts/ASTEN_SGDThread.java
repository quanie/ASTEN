package edu.uts;

import java.util.Iterator;
import java.util.Set;

/************************************************************************************************
 * Refactored by quan on 4/13/2016.
 ***********************************************************************************************/
public class ASTEN_SGDThread extends CTF_SGDThread {
    /**
     * Function ASTEN_SGDThread: initialize ASTEN_SGDThread class
     * @param numberOfTensor
     * @param tensorIdx
     * @param modeIdx
     * @param cTensor
     * @param pfactors
     * @param rank
     * @param numberOfThread
     * @param threadIdx
     */
    public ASTEN_SGDThread(int numberOfTensor,
                           int tensorIdx,
                           int modeIdx,
                           sparseCoupledTensor cTensor,
                           factor[][] pfactors,
                           int rank,
                           int numberOfThread,
                           int threadIdx) {
        nNumberOfTensor = numberOfTensor;
        spCTensor = cTensor;
        factors = pfactors;
        nTensorIdx = tensorIdx;
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
            else
                updateFactors();
        }
        else{
            if(nModeIdx==0){
                updateCoupledFactors();
            } // Coupled Mode
            else
                updateFactors();
        }
    }

    /************************************************************************************************
     * Function updateCoupledFactors
     ***********************************************************************************************/
    protected void updateCoupledFactors(){
        assert ((nTensorIdx==0&&nModeIdx+1<nNumberOfTensor)||(nTensorIdx>0&&nModeIdx==0));
        int nMainTensorIdx = nTensorIdx;
        int nMainTensorModeIdx = nModeIdx;
        int nCoupledTensorIdx = nModeIdx+1;
        int nCoupledTensorModeIdx = 0;

        if (nTensorIdx!=0){
            nMainTensorIdx = nTensorIdx;
            nMainTensorModeIdx = nModeIdx;
            nCoupledTensorIdx = 0;
            nCoupledTensorModeIdx = nTensorIdx-1;
        }

        int nLength = nEndIdx - nStartIdx;

        for(int rowIdx=0; rowIdx<nLength; rowIdx++)
        {
            Set setObservedEntry1 = spCTensor.getObservedEntryIdx(nMainTensorIdx, nMainTensorModeIdx, nStartIdx+rowIdx);
            int size1 = setObservedEntry1.size();

            int size = size1 + 1;

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
            double value =0;
            weight[cnt] = (double)1/factors[nMainTensorIdx][nMainTensorModeIdx].getLength();
            for (int r=0; r<nRank; r++){
                data[cnt][r] = 1;
                value += factors[nCoupledTensorIdx][nCoupledTensorModeIdx].get(nStartIdx+rowIdx, r);
            }
            label[cnt] = value;

            //Update parameters
            localFactor.setRow(rowIdx, optimize(data, label, parameter, weight));
        }
    }
}

