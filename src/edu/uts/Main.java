package edu.uts;

import java.io.IOException;
/************************************************************************************************
 * Refactored by quan on 4/7/2016.
 ***********************************************************************************************/

public class Main {
    //configuration parameters
    private static boolean DEBUG = true;
    private static final boolean CODING = false;
    private static int[][] gpnThread;         // number of Threads
    private static int gnRank;               //Decomposition rank
    private static int gnNumberOfDataSet;    //Number of dataset: 1: non-coupled/ >2: coupled
    private static int[][] gpnTensorLength;   //Length of input tensors piDim[tensorIdx][modeIdx]
    private static int[] gpnTensorMode;       //Modes of input tensors

    private static String[] gpsInputPath;     //input paths
    private static String gsOutputPath;      //output path

    private static double gdStepSize = 0.01;
    private static double gdMinDiff = 1.0E-6;
    private static long glRunningHour = 100;

    public static void main(String[] args) throws IOException{
	    // write your code here
        //1. Data preparation
        dataPreparation(args);

        //2. TensorFactorization
        if(gnNumberOfDataSet==1){
            TF Factorization = new TF();
            Factorization.setMinDiff(gdMinDiff);
            Factorization.setRank(gnRank);
            Factorization.setStepSize(gdStepSize);
            Factorization.setOutputPath(gsOutputPath);
            Factorization.setRunningTime(glRunningHour);
            Factorization.setCheckPoint(1);
            Factorization.factorize(gpnTensorMode[0], gpnTensorLength[0], gpnThread[0], gpsInputPath[0]);
        }
        else {
            ASTEN Factorization = new ASTEN();
            Factorization.setMinDiff(gdMinDiff);
            Factorization.setRank(gnRank);
            Factorization.setStepSize(gdStepSize);
            Factorization.setOutputPath(gsOutputPath);
            Factorization.setRunningTime(glRunningHour);
            Factorization.setCheckPoint(1);
            Factorization.setGradient("WSG");
            Factorization.factorize(gpnTensorMode, gpnTensorLength, gpnThread, gpsInputPath);
        }
    }
    /********************************************************************************************
     * Name: prepareData
     * Function: prepare environment from inputted parameters
     *              All tensors information is initialized here
     * Parameters:
     *      String[] args
     *
     * Return:  void
     *********************************************************************************************/
    private static void dataPreparation(String[] args) throws IOException
    {
        int i, j;
        int argsIdx;

        if(DEBUG)
        {
            for(i = 0; i < args.length; i++){
                System.out.println(i + " : " + args[i]);
            }
        }

        argsIdx=0;
        gnNumberOfDataSet = Integer.parseInt(args[argsIdx++]);
        if(DEBUG)
            System.out.println("Number of dataSet: " + gnNumberOfDataSet);

        gpnTensorMode = new int[gnNumberOfDataSet];
        gpnTensorLength = new int[gnNumberOfDataSet][];

        for(i=0;i<gnNumberOfDataSet;i++)
        {
            gpnTensorMode[i] = Integer.parseInt(args[argsIdx++]);	//mode of tensor1
            assert(gpnTensorMode[i]>1);

            gpnTensorLength[i] = new int[gpnTensorMode[i]];

            if(DEBUG)
                System.out.print("Tensor " + (i+1) + ":\n\tMode = " + gpnTensorMode[i] + "\n\tLength =");


            for(j=0;j<gpnTensorMode[i];j++)
            {
                gpnTensorLength[i][j] = Integer.parseInt(args[argsIdx++]);
                if(DEBUG)
                    System.out.print("\t" + gpnTensorLength[i][j]);
            }
            if(DEBUG)
                System.out.println("");
        }

        gpnThread = new int[gnNumberOfDataSet][];

        for(i=0; i<gnNumberOfDataSet; i++)
        {
            gpnThread[i] = new int[gpnTensorMode[i]];

            for(j=0; j<gpnTensorMode[i]; j++)
            {
                gpnThread[i][j] = Integer.parseInt(args[argsIdx++]);	//number of Node = number of split
            }

            if (i!=0)
                assert(gpnThread[i][0]==gpnThread[0][i-1]);
        }

        gnRank = Integer.parseInt(args[argsIdx++]);

        gpsInputPath = new String[gnNumberOfDataSet];

        for(i=0; i<gnNumberOfDataSet; i++)
            gpsInputPath[i] = args[argsIdx++];

        gsOutputPath = args[argsIdx++];

        //Stepsize
        gdStepSize = Double.parseDouble(args[argsIdx++]);

        if(DEBUG)
        {
            System.out.println("Rank: " + gnRank);
            System.out.println("Step Size: " + gdStepSize);

            for(i=0; i<gnNumberOfDataSet; i++)
                System.out.println("Tensor " + (i+1) +" path: " + gpsInputPath[i]);

            System.out.println("Output path: " + gsOutputPath);
        }

        //Prepare output
        gsOutputPath += "_" + "U~=A" + "_";
        for(i=0; i<gnNumberOfDataSet; i++)
        {
            if (gpsInputPath[i].contains("."))
            {
                String str = gpsInputPath[i].substring(0, gpsInputPath[i].lastIndexOf("."));
                gsOutputPath += str;
            }
            else
                gsOutputPath += gpsInputPath[i];
        }

        if (argsIdx<args.length)
        {
            gdMinDiff = (Double.parseDouble(args[argsIdx++]));
        }

        if (argsIdx<args.length)
        {
            glRunningHour = (Long.parseLong(args[argsIdx++]));
        }
        if (argsIdx<args.length)
        {
            DEBUG = (Integer.parseInt(args[argsIdx++])==0?false:true);
        }
        if(DEBUG)
            System.out.println("Done data preparation");
    }
}
