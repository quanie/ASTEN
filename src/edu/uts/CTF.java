package edu.uts;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/************************************************************************************************
 * Refactored by quan on 4/11/2016.
 ***********************************************************************************************/

public class CTF extends TF{
    protected factor[][] gArrFactors = null;       //variable to store all factors

    /************************************************************************************************
     * Function factorize: factorizes input tensor with their given file paths
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
                    if(modeIdx==0){
                        copyFactors(tensorIdx, 0, 0, tensorIdx-1);  //Copy Coupled Factor
                    }
                    else{
                        dMSE[tensorIdx] = updateFactors_T(nNumberOfTensor, tensorIdx, modeIdx, spCTensor, lTensorSize, nRank, dStepSize, threadNumber[tensorIdx][modeIdx]);
                    }
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
     * Function updateFactors_T: updates factors with multiThread
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
        CTF_SGDThread[] SGDt = new CTF_SGDThread[numberOfThread];
        Thread[]t = new Thread[numberOfThread];
        int threadIdx;

        for(threadIdx=0; threadIdx<numberOfThread; threadIdx++)
        {
            SGDt[threadIdx] = new CTF_SGDThread(numberOfTensor, tensorIdx, modeIdx, cTensor, gArrFactors, rank, numberOfThread, threadIdx);
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
    /************************************************************************************************
     * Function CopyFactors: copies B to A
     *
     * @param tensor1Idx
     * @param mode1Idx
     * @param tensor2Idx
     * @param mode2Idx
     *
     * Return: void
     ***********************************************************************************************/
    protected void copyFactors(int tensor1Idx, int mode1Idx, int tensor2Idx, int mode2Idx){
        for (int i=0; i<gArrFactors[tensor1Idx][mode1Idx].getLength(); i++){
            for (int r=0; r<gArrFactors[tensor1Idx][mode1Idx].getRank(); r++)
                gArrFactors[tensor1Idx][mode1Idx].setRow(i, gArrFactors[tensor2Idx][mode2Idx].getRow(i));
        }
    }
    /************************************************************************************************
     * Function createTensor: creates sparse Coupled Tensor
     *
     * @param numberOfTensor
     * @param tensorMode
     * @param tensorDimension
     * @param tensorFilePath
     *
     * Return: sparseCoupledTensor
     ***********************************************************************************************/
    protected sparseCoupledTensor createTensor(int numberOfTensor,
                                               int[] tensorMode,
                                               int[][] tensorDimension,
                                               String[] tensorFilePath) throws IOException {
        sparseCoupledTensor spCTensor = new sparseCoupledTensor(numberOfTensor, tensorMode, tensorDimension);

        final String DELIMITER = "\t";
        String line = "";

        for(int tensorIdx=0; tensorIdx<numberOfTensor; tensorIdx++)
        {
            BufferedReader fileReader = new BufferedReader(new FileReader(tensorFilePath[tensorIdx]));

            while ((line = fileReader.readLine()) != null) {
                String[] data = line.split(DELIMITER);

                assert(data.length == tensorMode[tensorIdx] + 1);
                double value = Double.parseDouble(data[tensorMode[tensorIdx]]);
                int[] index = new int[tensorMode[tensorIdx]];
                for (int i=0; i<tensorMode[tensorIdx]; i++)
                    index[i] = Integer.parseInt(data[i]);

                spCTensor.set(tensorIdx, index, value);
            }
            fileReader.close();
        }

        if (CODING){
            assert (spCTensor.checkObservedEntryIdx()==true);
        }

        return spCTensor;
    }
    /** ******************************************************************************************
     * Function initFullFactors: initializes full factors of the tensor
     *
     * @param initType 0: predefined value; 1: random with mean
     * @param initVal predefined value/ mean
     * @param tensorMode mode of the tensor => number of factors
     * @param tensorDimension dimension of each mode
     * @param rank second mode length
     *
     * Return: factor[][]
     * ********************************************************************************************/
    protected factor[][] initFullFactors(int initType,
                                         double initVal,
                                         int numberOfTensor,
                                         int[] tensorMode,
                                         int[][] tensorDimension,
                                         int rank){
        factor[][] arrFactors = new factor[numberOfTensor][];
        for(int tensorIdx=0; tensorIdx< numberOfTensor; tensorIdx++){
            arrFactors[tensorIdx] = new factor[tensorMode[tensorIdx]];

            for(int modeIdx=0;modeIdx<tensorMode[tensorIdx]; modeIdx++){
                arrFactors[tensorIdx][modeIdx] = initSingleFactor(initType, initVal, tensorDimension[tensorIdx][modeIdx], rank);
            }
        }
        return arrFactors;
    }
    /** ******************************************************************************************
     * Function saveLog: saves logging
     *
     * @param fileWriter
     * @param dMSE
     * @param dSMSE
     * @param lTime
     * @param nIter
     *
     * Return:  void
     * ********************************************************************************************/
    protected void saveLog(FileWriter fileWriter,
                           double[] dMSE,
                           double dSMSE,
                           long lTime,
                           int nIter) throws IOException {
        String strLoggingToWrite = "";

        if(nIter==0)
            strLoggingToWrite = "Iter" + "\tTime" + "\tMSE";

        strLoggingToWrite += "\n" + nIter +"\t" + lTime;

        for(int i=0; i<dMSE.length; i++)
            strLoggingToWrite += "\t" + dMSE[i];
        strLoggingToWrite += "\t" + dSMSE;

        fileWriter.append(strLoggingToWrite);
        fileWriter.flush();
    }

    /** ******************************************************************************************
     * Function saveFactors: saves factors
     *
     * Return: void
     * ********************************************************************************************/
    protected void saveFactors() throws IOException {
        File fOutputDir = new File(sOutputPath);

        for(int tensorIdx=0; tensorIdx<gArrFactors.length; tensorIdx++){
            for(int modeIdx=0; modeIdx<gArrFactors[tensorIdx].length; modeIdx++){
                //Create file for writing
                String str = "U_" + tensorIdx +"_" + modeIdx + ".txt";

                File fOutput = new File(fOutputDir, str);
                FileWriter fileWriterOutput = new FileWriter(fOutput);

                for(int rowIdx=0; rowIdx<gArrFactors[tensorIdx][modeIdx].getLength();rowIdx++){
                    String strOutputToWrite ="";
                    for(int col=0; col<gArrFactors[tensorIdx][modeIdx].getRank(); col++){
                        int[] index = {rowIdx,col};

                        if(col==0)
                            strOutputToWrite += gArrFactors[tensorIdx][modeIdx].get(index);
                        else
                            strOutputToWrite += "\t" + gArrFactors[tensorIdx][modeIdx].get(index);
                    }
                    strOutputToWrite +="\n";
                    //write output
                    fileWriterOutput.append(strOutputToWrite);
                    fileWriterOutput.flush();
                }
                fileWriterOutput.flush();
                fileWriterOutput.close();
            }
        }
    }
}
