package edu.uts;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/
public class TF {
    protected static final boolean CODING = true;     //debug at coding stage
    protected static final boolean DEBUG = false;     //debug at coding stage

    protected double dMinDiff = 1.0E-1;     //Stopping condition - Error difference
    protected double dRunTime = 100;        //Stopping condition - running hour
    protected double dStepSize = 0.01;
    protected int nRank = 3;                //default decomposition rank
    protected String sOutputPath = "";      //output path
    protected int nCheckPointIter = 10;     //CheckPoint after each 10 iterations

    protected String sGradient = "LSG";     //LSG: leastSquaredGradient; WSG: weightedSquaredGradient
    protected String sUpdater = "Simple";   //Simple: SimpleUpdater; L1: L1Updater; L2: L2Updater

    private factor[] gArrFactors = null;       //variable to store all factors

    /************************************************************************************************
     * Set the initial step size of SGD for the first step. Default 1.0.
     * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
     ***********************************************************************************************/
    public void setStepSize(double step){this.dStepSize = step;}
    public void setMinDiff(double minDiff){this.dMinDiff = minDiff;}
    public void setRunningTime(double runTime){this.dRunTime = runTime;}
    public void setOutputPath(String path){this.sOutputPath = path;}
    public void setRank(int rank){this.nRank = rank;}
    public int getRank(){return nRank;}
    public void setCheckPoint(int checkPointIter){this.nCheckPointIter = checkPointIter;}
    public void setGradient(String s){sGradient = s;}
    public void setUpdater(String s){sUpdater = s;}

    /************************************************************************************************
     * Function factorize: runs Tensor Factorization on the given tensor file
     *
     * @param tensorMode Mode of the tensor.
     * @param tensorDimension Dimension of the tensor
     * @param tensorFilePath Path of the tensor
     ***********************************************************************************************/
    public void factorize(int tensorMode,
                          int[] tensorDimension,
                          int[] threadNumber,
                          String tensorFilePath) throws IOException {
        double dPreLoss = 1.0E30;
        double dMSEDiff = 0;
        double dMSE = 0;

        long lStartTime, lEndTime, lTimeLapsed = 0;
        int nIter = 0;

        boolean bIsConverged = false;
        FileWriter fileWriter = initLogging();
        //0. Double check parameters
        assert (tensorDimension.length == threadNumber.length);
        for(int i=0; i<tensorDimension.length;i++){
            assert (threadNumber[i]<=tensorDimension[i]);
        }

        //1. Create tensor
        sparseTensor spTensor = createTensor(tensorMode, tensorDimension, tensorFilePath);
        long lTensorSize = spTensor.getObservedEntryNumber();
        System.out.println("TensorSize: " + lTensorSize);

        //2. Init Factors
        gArrFactors = initFullFactors(0, 1, tensorMode, tensorDimension, nRank);

        //3. Factorization using SGD
        //Get start time
        lStartTime = System.currentTimeMillis() / 1000L;

        //while(nIter<1){
        while(!bIsConverged && (lTimeLapsed/3600<dRunTime)) {
            //3.1 Update Factors one by one
            for (int modeIdx=0; modeIdx<tensorMode; modeIdx++){
                dMSE = updateFactors_T(modeIdx, spTensor, lTensorSize, nRank, dStepSize, threadNumber[modeIdx]);
            }

            //3.2 Compute Loss
            dMSEDiff = dPreLoss - dMSE;
            double relfit = dMSEDiff/dPreLoss;
            if(relfit<dMinDiff||Double.isNaN(relfit)||Double.isInfinite(relfit)){
                bIsConverged = true;
                saveLog(fileWriter, dMSE, lTimeLapsed, nIter);
                saveFactors();
                finalizeLogging(fileWriter);
            }
            dPreLoss = dMSE;

            lEndTime = System.currentTimeMillis() / 1000L;
            lTimeLapsed = lEndTime - lStartTime;

            System.out.println("Time: " + lTimeLapsed + "\tMSE: "+ dMSE +"\tDiff: " + dMSEDiff + "\tDiff/Pre: " + relfit);

            //Logging
            if(!bIsConverged&&(nIter%nCheckPointIter==0)){
                saveLog(fileWriter, dMSE, lTimeLapsed, nIter);
                saveFactors();
            }

            nIter++;
        }
    }
    /** ******************************************************************************************
     * Function updateFactors_T: updates factors with multiThread
     *
     * @param modeIdx factor at modexIdx to be updated
     * @param tensor tensor
     * @param tensorSize size of the tensor
     * @param rank rank of the tensor decomposition
     * @param stepSize learning stepsize
     *
     * Return: Loss
     * ********************************************************************************************/
    protected double updateFactors_T(int modeIdx,
                                   sparseTensor tensor,
                                   long tensorSize,
                                   int rank,
                                   double stepSize,
                                   int numberOfThread){
        TF_SGDThread[] SGDt = new TF_SGDThread[numberOfThread];
        Thread[]t = new Thread[numberOfThread];
        int threadIdx;

        for(threadIdx=0; threadIdx<numberOfThread; threadIdx++)
        {
            SGDt[threadIdx] = new TF_SGDThread(modeIdx, tensor, gArrFactors, rank, numberOfThread, threadIdx);
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
                gArrFactors[modeIdx].setRow(rowIdx, localFactor.getRow(rowIdx-nStartIdx));
            }
        }

        return dLoss;
    }
    /** ******************************************************************************************
     * Function createTensor: creates tensor from input file
     *
     * @param tensorMode mode of the tensor
     * @param tensorDimension dimension of each mode
     * @param tensorFilePath input filename
     *
     * Return: sparseTensor
     * ********************************************************************************************/
    private sparseTensor createTensor(int tensorMode,
                                        int[] tensorDimension,
                                        String tensorFilePath) throws IOException {
        sparseTensor spTensor = new sparseTensor(tensorMode, tensorDimension);

        final String DELIMITER = "\t";
        String line = "";
        BufferedReader fileReader = new BufferedReader(new FileReader(tensorFilePath));
        int count = 0;

        while ((line = fileReader.readLine()) != null) {
            count++;
            String[] data = line.split(DELIMITER);

            assert(data.length == tensorMode + 1);
            double value = Double.parseDouble(data[tensorMode]);
            int[] index = new int[tensorMode];
            for (int i=0; i<tensorMode; i++)
                index[i] = Integer.parseInt(data[i]);

            spTensor.set(index, value);
        }
        fileReader.close();

        if (CODING){
            assert (spTensor.checkObservedEntryIdx()==true);
            assert (count == spTensor.getObservedEntryNumber());
        }

        return spTensor;
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
     * Return: factor[]
     * ********************************************************************************************/
    protected factor[] initFullFactors(int initType,
                                       double initVal,
                                       int tensorMode,
                                       int[] tensorDimension,
                                       int rank){
        factor[] arrFactors = new factor[tensorMode];
        for(int i=0;i<tensorMode; i++){
            arrFactors[i] = initSingleFactor(initType, initVal, tensorDimension[i], rank);
        }

        return arrFactors;
    }
    /** ******************************************************************************************
     * Function initSingleFactor: initializes a single factor
     *
     * @param initType 0: predefined value; 1: random with mean
     * @param initVal predefined value/ mean
     * @param length first mode length
     * @param rank second mode length
     *
     * Return: factor
     * ********************************************************************************************/
    protected factor initSingleFactor(int initType, double initVal, int length, int rank){
        factor factor0 = new factor(length, rank);
        if (initType == 0) //Init with predefined value
            factor0.init(initVal);
        else //Init with random value
            factor0.reset(initVal);

        return factor0;
    }

    /** ******************************************************************************************
     * Function saveFactors: saves factors
     *
     * Return: void
     * ********************************************************************************************/
    protected void saveFactors() throws IOException {
        File fOutputDir = new File(sOutputPath);

        for(int j=0; j<gArrFactors.length; j++){
            //Create file for writing
            String str = "U_" + j + ".txt";

            File fOutput = new File(fOutputDir, str);
            FileWriter fileWriterOutput = new FileWriter(fOutput);

            for(int row=0; row<gArrFactors[j].getLength();row++){
                String strOutputToWrite ="";
                for(int col=0; col<gArrFactors[j].getRank(); col++){
                    int[] index = {row,col};

                    if(col==0)
                        strOutputToWrite += gArrFactors[j].get(index);
                    else
                        strOutputToWrite += "\t" + gArrFactors[j].get(index);
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

    /** ******************************************************************************************
     * Function initLogging: initializes Logging
     *
     * Return:  FileWriter
     * ********************************************************************************************/
    protected FileWriter initLogging() throws IOException {
        File fOutputDir = new File(sOutputPath);
        fOutputDir.mkdirs();

        File fLogging = new File(fOutputDir, "logging.txt");
        return new FileWriter(fLogging);
    }
    /** ******************************************************************************************
     * Function saveLog: saves logging
     *
     * @param fileWriter
     * @param dMSE
     * @param lTime
     * @param nIter
     *
     * Return:  void
     * ********************************************************************************************/
    protected void saveLog(FileWriter fileWriter, double dMSE, Long lTime, int nIter) throws IOException {
        String strLoggingToWrite = "";

        if(nIter==0)
            strLoggingToWrite = "Iter" + "\tTime" + "\tMSE";

        strLoggingToWrite += "\n" + nIter +"\t" + lTime;
        strLoggingToWrite += "\t" + dMSE;

        fileWriter.append(strLoggingToWrite);
        fileWriter.flush();
    }
    /** ******************************************************************************************
     * Function finalizeLogging: finalizes logging
     *
     * @param fileWriter
     *
     * Return:  void
     * ********************************************************************************************/
    protected void finalizeLogging(FileWriter fileWriter) throws IOException {
        fileWriter.close();
    }
}
