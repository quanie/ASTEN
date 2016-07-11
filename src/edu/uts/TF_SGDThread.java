package edu.uts;

import java.util.Iterator;
import java.util.Set;

/************************************************************************************************
 * Refactored by quan on 4/8/2016.
 ***********************************************************************************************/
public class TF_SGDThread extends SGDThread{
    private final sparseTensor spTensor;
    private final factor[] factors;
    protected long lDataSize = 1;
    protected int nModeIdx;
    protected int nRank;
    protected int nThreadIdx;
    protected int nNumberOfThread;

    /************************************************************************************************
     * Function setDataSize: sets tensor Size
     *
     * @param dataSize
     ***********************************************************************************************/
    public void setDataSize(long dataSize){lDataSize = dataSize;}

    /************************************************************************************************
     * Function TF_SGDThread: initializes TF_SGDThread class
     ***********************************************************************************************/
    TF_SGDThread(){
        nModeIdx = -1;
        this.spTensor = null;
        this.factors = null;
        nRank = 0;
        nNumberOfThread = 0;
        nThreadIdx = 0;
        assert (nModeIdx!=-1);
    }

    /************************************************************************************************
     * Function TF_SGDThread: initializes TF_SGDThread class with the following parameters
     *
     * @param modeIdx
     * @param tensor
     * @param factors
     * @param rank
     * @param numberOfThread
     * @param threadIdx
     ***********************************************************************************************/
    TF_SGDThread(int modeIdx,
              sparseTensor tensor,
              factor[] factors,
              int rank,
              int numberOfThread,
              int threadIdx){
        nModeIdx = modeIdx;
        this.spTensor = tensor;
        this.factors = factors;
        nRank = rank;
        nNumberOfThread = numberOfThread;
        nThreadIdx = threadIdx;

        int nLength;
        //Create localFactor data
        if(nThreadIdx<nNumberOfThread-1) {
            nLength = (int) Math.floor(1.0 * factors[modeIdx].getLength() / nNumberOfThread);
            nStartIdx = nThreadIdx * nLength;
        }
        else//last part => get all the remaining
        {
            nLength = factors[modeIdx].getLength() - nThreadIdx * (int) Math.floor(1.0 * factors[modeIdx].getLength() / nNumberOfThread);
            nStartIdx = nThreadIdx * (int) Math.floor(1.0 * factors[modeIdx].getLength() / nNumberOfThread);
        }
        nEndIdx = nStartIdx + nLength;

        assert(nEndIdx<=factors[modeIdx].getLength());
        localFactor = new factor(nLength, nRank);

        //Init localFactor
        for(int rowIdx=0; rowIdx<nLength; rowIdx++)
            localFactor.setRow(rowIdx, factors[modeIdx].getRow(rowIdx + nStartIdx));

        //Reset dLoss =0
        dLoss = 0;
    }

    /************************************************************************************************
     * Function optimize: optimizes factors
     *
     * @param data
     * @param label
     * @param parameter
     * @param weight
     *
     * @return double[]
     ***********************************************************************************************/
    @Override
    protected double[] optimize(double[][] data, double[] label, double[] parameter, double[] weight) {
        double[] dGradientSum = new double[nRank];
        double[] newParameter = null;

        for (int r = 0; r < nRank; r++)
            dGradientSum[r] = 0;
        /**
         * For the first iteration, the regVal will be initialized as sum of weight squares
         * if it's L2 updater; for L1 updater, the same logic is followed.
         */
        double regVal = usedUpdater.compute(parameter, dGradientSum, dStepSize, dRegParam);

        for (int iter=0; iter<nNumIteration; iter++) {
            for (int i = 0; i < label.length; i++) {
                double[] dGradient = usedGradient.compute(data[i], label[i], parameter, weight[i]);

                for (int r = 0; r < nRank; r++)
                    dGradientSum[r] += dGradient[r];

                dLoss+=dGradient[nRank];
            }

            /**
             * lossSum is computed using the weights from the previous iteration
             * and regVal is the regularization value computed in the previous iteration as well.
             */
            regVal = usedUpdater.compute(parameter, dGradientSum, dStepSize, regVal);
            newParameter = usedUpdater.getUpdatedParameter();
        }

        return newParameter;
    }

    /************************************************************************************************
     * Function run: runs the updates
     ***********************************************************************************************/
    @Override
    public void run() {
        int nLength = nEndIdx - nStartIdx;

        for(int rowIdx=0; rowIdx<nLength; rowIdx++)
        {
            Set setObservedEntry = spTensor.getObservedIdx(nModeIdx, nStartIdx+rowIdx);
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

                label[cnt] = spTensor.get(index);

                for (int r=0; r<nRank; r++) {
                    double dotValue = 1;

                    for (int j=0; j<pEntryIdx.length; j++){
                        if (j!=nModeIdx)
                            dotValue*=factors[j].get(index[j], r);
                    }

                    data[cnt][r] = dotValue;
                }
                weight[cnt] = (double)1/lDataSize;

                cnt++;
            }

            //Update parameters
            localFactor.setRow(rowIdx, optimize(data, label, parameter, weight));
        }
    }
}