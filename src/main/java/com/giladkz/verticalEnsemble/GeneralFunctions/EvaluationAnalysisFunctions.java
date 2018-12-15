package com.giladkz.verticalEnsemble.GeneralFunctions;

import com.giladkz.verticalEnsemble.Data.EvaluationInfo;
import com.giladkz.verticalEnsemble.Data.EvaluationPerIteraion;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class EvaluationAnalysisFunctions {
    /**
     * Combines the results of several classifiers by averaging
     * @param evaluationResultsPerPartition
     * @param numOfClasses
     * @return
     */
    public static double[][] calculateAverageClassificationResults(HashMap<Integer,EvaluationInfo> evaluationResultsPerPartition, int numOfClasses) {
        double[][] resultsToReturn = new double[ evaluationResultsPerPartition.get(0).getScoreDistributions().length][numOfClasses];
        for (int partition : evaluationResultsPerPartition.keySet()) {
            for (int i=0; i<evaluationResultsPerPartition.get(partition).getScoreDistributions().length; i++) {
                for (int j=0; j<numOfClasses; j++) {
                    resultsToReturn[i][j] += evaluationResultsPerPartition.get(partition).getScoreDistributions()[i][j];
                }
            }
        }
        //now we normalize
        for (int i=0; i<resultsToReturn.length; i++) {
            for (int j=0; j<numOfClasses; j++) {
                 resultsToReturn[i][j] = resultsToReturn[i][j] / numOfClasses;
            }
        }
        return normalizeClassificationResults(resultsToReturn);
    }

    /**
     * combines the results of several classifiers (partitions) by multiplication. After the multiplication
     * is complete, each confidence score is divided by the ratio of the class in the original labeled dataset
     * @param evaluationResultsPerPartition
     * @param numOfClasses
     * @param classRatios
     * @return
     */
    public static double[][] calculateMultiplicationClassificationResults(HashMap<Integer,EvaluationInfo> evaluationResultsPerPartition,
                                                                   int numOfClasses, HashMap<Integer, Double> classRatios) {
        double[][] resultsToReturn = new double[ evaluationResultsPerPartition.get(0).getScoreDistributions().length][numOfClasses];
        for (int partition : evaluationResultsPerPartition.keySet()) {
            for (int i=0; i<evaluationResultsPerPartition.get(partition).getScoreDistributions().length; i++) {
                for (int j=0; j<numOfClasses; j++) {
                    if (partition == 0) { //if its the first partition we analyze, simply assign the value
                        resultsToReturn[i][j] = evaluationResultsPerPartition.get(partition).getScoreDistributions()[i][j];
                    }
                    else { //otherwise, multiply
                        resultsToReturn[i][j] *= evaluationResultsPerPartition.get(partition).getScoreDistributions()[i][j];
                    }
                }
            }
        }
        for (int i=0; i<evaluationResultsPerPartition.get(0).getScoreDistributions().length; i++) {
            for (int j = 0; j < numOfClasses; j++) {
                resultsToReturn[i][j] = resultsToReturn[i][j] / classRatios.get(j);
            }
        }
        return normalizeClassificationResults(resultsToReturn);
    }

    /**
     * Normalizes the classifiaction results for each instance
     * @param results
     * @return
     */
    public static double[][] normalizeClassificationResults(double[][] results) {
        for (int i=0; i<results.length; i++) {
            double sum = 0;
            for (int j=0; j<results[0].length; j++) {
                sum += results[i][j];
            }
            for (int j=0; j<results[0].length; j++) {
                results[i][j] = results[i][j]/sum;
            }
        }
        return results;
    }

    public static double[] calculatePercentileClassificationResults(HashMap<Integer, double[][]> iterationScoreDistPerPartition, int targetClassIndex, int instancePos){
        double[] resultsToReturn = new double[iterationScoreDistPerPartition.keySet().size()];
        for (int partition : iterationScoreDistPerPartition.keySet()) {
            double[] scoreToPercentile = new double[iterationScoreDistPerPartition.get(partition).length];
            for (int instance = 0; instance < iterationScoreDistPerPartition.get(partition).length; instance++){
                double instanceScore = iterationScoreDistPerPartition.get(partition)[instance][targetClassIndex];
                if (instanceScore == 0.0){
                    instanceScore = 0.0001;
                }
                scoreToPercentile[instance] = instanceScore;
            }
            Percentile p = new Percentile();

/*            double[] normalizedScoreToPercentile = new double[scoreToPercentile.length];
            double max = Arrays.stream(normalizedScoreToPercentile).max().getAsDouble();
            double min = Arrays.stream(normalizedScoreToPercentile).min().getAsDouble();
            for (int i = 0; i < scoreToPercentile.length; i++) {
                normalizedScoreToPercentile[i] = ((scoreToPercentile[i] - min)/(max - min))*100;
            }
            double normValToCals = ((scoreToPercentile[instancePos]- min)/(max - min))*100;
            p.setData(normalizedScoreToPercentile);
            resultsToReturn[partition] = p.evaluate(normValToCals);*/

            p.setData(scoreToPercentile);
            resultsToReturn[partition] = p.evaluate(scoreToPercentile[instancePos]);

        }
        return resultsToReturn;
    }

}
