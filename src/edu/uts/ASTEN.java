package edu.uts;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/************************************************************************************************
 * Refactored by quan on 4/13/2016.
 ***********************************************************************************************/
public class ASTEN extends CTF{
    /************************************************************************************************
     * Function factorize: factorizes input tensors with their given file paths
     *
     * @param tensorMode Mode of the tensor.
     * @param tensorDimension Dimension of the tensor
     * @param tensorFilePath Path of the tensor
     ***********************************************************************************************/
    public void factorize(int[] tensorMode,
                          int[][] tensorDimension,
                          int[][] threadNumber,
                          String[] tensorFilePath) throws IOException {
        double dPreLoss = 1.0E30;
        double dMSEDiff = 0;
        int nNumberOfTensor = tensorMode.length;

        double[] dMSE = new double[nNumberOfTensor];

        long lStartTime, lEndTime, lTimeLapsed = 0;
        int nIter = 0;

        boolean bIsConverged = false;
        FileWriter fileWriter = initLogging();
        //0. Double check parameters

        assert (tensorDimension.length == threadNumber.length);
        for(int tensorIdx=0; tensorIdx<tensorDimension.length; tensorIdx++){
            assert (tensorDimension[tensorIdx].length == threadNumber[tensorIdx].length);
            for(int modeIdx = 0; modeIdx<tensorMode[tensorIdx]; modeIdx++)
                assert (threadNumber[tensorIdx][modeIdx]<=tensorDimension[tensorIdx][modeIdx]);
        }

        //1. Create tensor
        sparseCoupledTensor spCTensor = createTensor(nNumberOfTensor, tensorMode, tensorDimension, tensorFilePath);
        long[] lTensorSize = spCTensor.getObservedEntryNumber();
        for(int tensorIdx=0; tensorIdx< nNumberOfTensor; tensorIdx++)
            System.out.println("Tensor " + tensorIdx + " Size: " + lTensorSize[tensorIdx]);

        //2. Init Factors
        gArrFactors = initFullFactors(0, 1, nNumberOfTensor, tensorMode, tensorDimension, nRank);

        //3. Factorization using SGD
        //Get start time
        lStartTime = System.currentTimeMillis() / 1000L;

        //while(nIter<1){
        while(!bIsConverged && (lTimeLapsed/3600<dRunTime)) {
            //3.1 Update Main Tensor
            for(int modeIdx=0; modeIdx<tensorMode[0]; modeIdx++){
                dMSE[0] = updateFactors_T(nNumberOfTensor, 0, modeIdx, spCTensor, lTensorSize, nRank, dStepSize, threadNumber[0][modeIdx]);
            }

            //3.2 Update Coupled Tensors
            for(int tensorIdx=1; tensorIdx<nNumberOfTensor; tensorIdx++){
                for(int modeIdx=0; modeIdx<tensorMode[tensorIdx]; modeIdx++){
                    dMSE[tensorIdx] = updateFactors_T(nNumberOfTensor, tensorIdx, modeIdx, spCTensor, lTensorSize, nRank, dStepSize, threadNumber[tensorIdx][modeIdx]);
                }
            }


            //3.2 Compute Loss
            double dSMSE = 0;
            for (int i=0;i<nNumberOfTensor; i++)
                dSMSE += dMSE[i];

            dMSEDiff = dPreLoss - dSMSE;
            double relfit = dMSEDiff/dPreLoss;
            if(relfit<dMinDiff||Double.isNaN(relfit)||Double.isInfinite(relfit)){
                bIsConverged = true;
                saveLog(fileWriter, dMSE, dSMSE, lTimeLapsed, nIter);
                saveFactors();
                finalizeLogging(fileWriter);
            }
            dPreLoss = dSMSE;

            lEndTime = System.currentTimeMillis() / 1000L;
            lTimeLapsed = lEndTime - lStartTime;

            System.out.print("Time: " + lTimeLapsed);
            for(int tensorIdx=0; tensorIdx<nNumberOfTensor; tensorIdx++)
                System.out.print("\tMSE: "+ dMSE[tensorIdx]);
            System.out.println("\tSMSE: " + dSMSE + "\tDiff: " + dMSEDiff + "\tDiff/Pre: " + relfit);

            //Logging
            if(!bIsConverged&&(nIter%nCheckPointIter==0)){
                saveLog(fileWriter, dMSE, dSMSE, lTimeLapsed, nIter);
                saveFactors();
            }

            nIter++;
        }
    }
    /** ******************************************************************************************
     * Function updateFactors_T: update factors with multiThread
     *
     * @param tensorIdx tensorIdx
     * @param modeIdx factor at modexIdx to be updated
     * @param cTensor tensor
     * @param tensorSize size of the tensor
     * @param rank rank of the tensor decomposition
     * @param stepSize learning stepsize
     *
     * Return: Loss
     * ********************************************************************************************/
    protected double updateFactors_T(int numberOfTensor,
                                     int tensorIdx,
                                     int modeIdx,
                                     sparseCoupledTensor cTensor,
                                     long[] tensorSize,
                                     int rank,
                                     double stepSize,
                                     int numberOfThread){
        ASTEN_SGDThread[] SGDt = new ASTEN_SGDThread[numberOfThread];
        Thread[]t = new Thread[numberOfThread];
        int threadIdx;

        for(threadIdx=0; threadIdx<numberOfThread; threadIdx++)
        {
            SGDt[threadIdx] = new ASTEN_SGDThread(numberOfTensor, tensorIdx, modeIdx, cTensor, gArrFactors, rank, numberOfThread, threadIdx);
            SGDt[threadIdx].setDataSize(tensorSize);
            SGDt[threadIdx].setStepSize(stepSize);
            switch (sGradient){
                case "LSG": SGDt[threadIdx].setGradient(new leastSquaredGradient());
                    break;
                case "WSG": SGDt[threadIdx].setGradient(new weightedSquaredGradient());
                    break;
                default:
                    assert (sGradient=="LSG" || sGradient =="WSG");
                    break;
            }
            switch (sUpdater){
                case "Simple": SGDt[threadIdx].setUpdater(new simpleUpdater());
                    break;
                case "L1": SGDt[threadIdx].setUpdater(new L1Updater());
                    break;
                case "L2": SGDt[threadIdx].setUpdater(new L2Updater());
                    break;
                default:
                    assert (sUpdater=="Simple" || sUpdater=="L1" || sUpdater=="L2");
                    break;
            }

            t[threadIdx] = new Thread(SGDt[threadIdx], String.valueOf(threadIdx));
            t[threadIdx].start();
        }

        //Wait until all threads finish
        for(threadIdx=0; threadIdx<numberOfThread; threadIdx++)
        {
            try {
                t[threadIdx].join();
            } catch (InterruptedException ex) {
                Logger.getLogger(TF.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        double dLoss = 0;
        //Update factor
        for(threadIdx=0; threadIdx<numberOfThread; threadIdx++){
            int nStartIdx = SGDt[threadIdx].getStartIdx();
            int nEndIdx = SGDt[threadIdx].getEndIdx();
            factor localFactor = SGDt[threadIdx].getLocalFactor();
            dLoss += SGDt[threadIdx].getLoss();

            for (int rowIdx=nStartIdx; rowIdx<nEndIdx; rowIdx++) {
                gArrFactors[tensorIdx][modeIdx].setRow(rowIdx, localFactor.getRow(rowIdx-nStartIdx));
            }
        }

        return dLoss;
    }
}
