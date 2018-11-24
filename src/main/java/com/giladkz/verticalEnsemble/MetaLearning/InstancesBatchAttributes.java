package com.giladkz.verticalEnsemble.MetaLearning;

import com.giladkz.verticalEnsemble.Data.AttributeInfo;
import com.giladkz.verticalEnsemble.Data.Column;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Data.EvaluationPerIteraion;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.math3.stat.inference.ChiSquareTest;


import java.util.*;

/**
 * Created by giladkatz on 5/9/18.
 */
public class InstancesBatchAttributes {

    private List<Integer> numOfIterationsBackToAnalyze = Arrays.asList(1,3,5,10);

    /**
     *
     * @param trainingDataset
     * @param testDataset
     * @param currentIterationIndex
     * @param evaluationResultsPerSetAndInteration
     * @param unifiedDatasetEvaulationResults
     * @param targetClassIndex
     * @param instancesBatchPos - arrayList of all instances positions
     * @param assignedLabels - structure: <instancePos, assignedLabel>
     * @param properties
     * @return
     */
    public TreeMap<Integer, AttributeInfo> getInstancesBatchAssignmentMetaFeatures(
                Dataset trainingDataset, Dataset testDataset, int currentIterationIndex
                , TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
                , EvaluationPerIteraion unifiedDatasetEvaulationResults, int targetClassIndex
                , ArrayList<Integer> instancesBatchPos, HashMap<Integer, Integer> assignedLabels, Properties properties) {


        TreeMap<Integer, AttributeInfo> instanceAttributesToReturn = new TreeMap<>();

        //batch size
        int batchSize = instancesBatchPos.size();
        AttributeInfo batchSizeAttr = new AttributeInfo
                ("batchSize", Column.columnType.Numeric, batchSize, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchSizeAttr);

        //instances labeled as 0
        int countLabel0 = 0;
        for (Integer instancePos: assignedLabels.keySet()) {
            if (assignedLabels.get(instancePos) == 0){
                countLabel0++;
            }
        }
        AttributeInfo batchLabel0 = new AttributeInfo
                ("batchLabel0", Column.columnType.Numeric, countLabel0, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchLabel0);

        //batch score distribution
        //per partition
        DescriptiveStatistics batchScoreDistLabel0 = new DescriptiveStatistics();
        DescriptiveStatistics batchScoreDistLabel1 = new DescriptiveStatistics();
        HashMap<Integer, double[]> distanceBatchPairsPerPartition = new HashMap<>();
        HashMap<Integer, HashMap<Integer, Double>> scorePerInstancePerPartition = new HashMap<>();
        for (Integer partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            //collect scores for statistics
            DescriptiveStatistics batchScoreDistPerPartition = new DescriptiveStatistics();
            HashMap<Integer, Double> scorePerInstanceTemp = new HashMap<>();
            for (Integer instancePos: assignedLabels.keySet()) {
                double instanceScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions()[instancePos][targetClassIndex];
                batchScoreDistPerPartition.addValue(instanceScore);
                scorePerInstanceTemp.put(instancePos, instanceScore);
                //collect scores for statistics per label
                if (assignedLabels.get(instancePos) == 0){
                    batchScoreDistLabel0.addValue(evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions()[instancePos][assignedLabels.get(instancePos)]);
                }
                else if (assignedLabels.get(instancePos) == 1){
                    batchScoreDistLabel1.addValue(evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions()[instancePos][assignedLabels.get(instancePos)]);
                }
            }
            scorePerInstancePerPartition.put(partitionIndex, scorePerInstanceTemp);
            //max
            AttributeInfo batchScoreMax = new AttributeInfo
                    ("batchScoreMax" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getMax(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMax);
            //min
            AttributeInfo batchScoreMin = new AttributeInfo
                    ("batchScoreMin" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getMin(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMin);
            //mean
            AttributeInfo batchScoreMean = new AttributeInfo
                    ("batchScoreMean" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getMean(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMean);
            //std
            AttributeInfo batchScoreStd = new AttributeInfo
                    ("batchScoreStd" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getStandardDeviation(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreStd);
            //p-50
            AttributeInfo batchScoreMedian = new AttributeInfo
                    ("batchScoreMedian" + partitionIndex, Column.columnType.Numeric, batchScoreDistPerPartition.getPercentile(50), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMedian);

            //pair distance
            double[] distancePerPair = new double[instanceAttributesToReturn.size()];
            int distancePerPairCounter = 0;
            for (int i = 0; i <  batchScoreDistPerPartition.getValues().length - 1; i++) {
                for (int j = i+1; j < batchScoreDistPerPartition.getValues().length; j++) {
                    distancePerPair[distancePerPairCounter] = Math.abs(batchScoreDistPerPartition.getValues()[i] - batchScoreDistPerPartition.getValues()[j]);
                    distancePerPairCounter++;
                }
            }
            distanceBatchPairsPerPartition.put(partitionIndex, distancePerPair);
        }

        //statistics per labels
        //label 0
        //max
        AttributeInfo batchScoreMaxLabel0 = new AttributeInfo
                ("batchScoreMaxLabel0", Column.columnType.Numeric, batchScoreDistLabel0.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMaxLabel0);
        //min
        AttributeInfo batchScoreMinLabel0 = new AttributeInfo
                ("batchScoreMinLabel0", Column.columnType.Numeric, batchScoreDistLabel0.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMinLabel0);
        //mean
        AttributeInfo batchScoreMeanLabel0 = new AttributeInfo
                ("batchScoreMeanLabel0", Column.columnType.Numeric, batchScoreDistLabel0.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMeanLabel0);
        //std
        AttributeInfo batchScoreStdLabel0 = new AttributeInfo
                ("batchScoreStdLabel0", Column.columnType.Numeric, batchScoreDistLabel0.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreStdLabel0);
        //p-50
        AttributeInfo batchScoreMedianLabel0 = new AttributeInfo
                ("batchScoreMedianLabel0", Column.columnType.Numeric, batchScoreDistLabel0.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMedianLabel0);
        //label 1
        //max
        AttributeInfo batchScoreMaxLabel1 = new AttributeInfo
                ("batchScoreMaxLabel1", Column.columnType.Numeric, batchScoreDistLabel1.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMaxLabel1);
        //min
        AttributeInfo batchScoreMinLabel1 = new AttributeInfo
                ("batchScoreMinLabel1", Column.columnType.Numeric, batchScoreDistLabel1.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMinLabel1);
        //mean
        AttributeInfo batchScoreMeanLabel1 = new AttributeInfo
                ("batchScoreMeanLabel1", Column.columnType.Numeric, batchScoreDistLabel1.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMeanLabel1);
        //std
        AttributeInfo batchScoreStdLabel1 = new AttributeInfo
                ("batchScoreStdLabel1", Column.columnType.Numeric, batchScoreDistLabel1.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreStdLabel1);
        //p-50
        AttributeInfo batchScoreMedianLabel1 = new AttributeInfo
                ("batchScoreMedianLabel1", Column.columnType.Numeric, batchScoreDistLabel1.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchScoreMedianLabel1);

        //delta between partition - per instance
        DescriptiveStatistics batchDeltaScore = new DescriptiveStatistics();
        for (Integer instancePos: scorePerInstancePerPartition.get(0).keySet()) {
            double instanceDelta = Math.abs(scorePerInstancePerPartition.get(0).get(instancePos)
                    - scorePerInstancePerPartition.get(1).get(instancePos));
            batchDeltaScore.addValue(instanceDelta);
        }
        //max
        AttributeInfo batchDeltaScoreMax = new AttributeInfo
                ("batchDeltaScoreMax", Column.columnType.Numeric, batchDeltaScore.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMax);
        //min
        AttributeInfo batchDeltaScoreMin = new AttributeInfo
                ("batchDeltaScoreMin", Column.columnType.Numeric, batchDeltaScore.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMin);
        //mean
        AttributeInfo batchDeltaScoreMean = new AttributeInfo
                ("batchDeltaScoreMean", Column.columnType.Numeric, batchDeltaScore.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMean);
        //std
        AttributeInfo batchDeltaScoreStd = new AttributeInfo
                ("batchDeltaScoreStd", Column.columnType.Numeric, batchDeltaScore.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreStd);
        //p-50
        AttributeInfo batchDeltaScoreMedian = new AttributeInfo
                ("batchDeltaScoreMedian", Column.columnType.Numeric, batchDeltaScore.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchDeltaScoreMedian);

        //distance statistics
        //use: distanceBatchPairsPerPartition


        //paired t-test
        TTest batchTtest = new TTest();
        DescriptiveStatistics batchPartitionIterationBackTtestScore = new DescriptiveStatistics();
        DescriptiveStatistics batchIterationBackTtestScore = new DescriptiveStatistics();
        for (Integer partitionIndex : evaluationResultsPerSetAndInteration.keySet()){
            double[][] currentIterScoreDist = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
            double[] currentIterationTargetClassScoreDistribution = new double[currentIterScoreDist.length];
            for (int i = 0; i < currentIterScoreDist.length; i++) {
                currentIterationTargetClassScoreDistribution[i] = currentIterScoreDist[i][targetClassIndex];
            }
            for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
                if (currentIterationIndex >= numOfIterationsBack) {
                    double[][] prevIterScoreDist = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
                    double[] prevIterationTargetClassScoreDistribution = new double[prevIterScoreDist.length];
                    for (int j = 0; j < currentIterScoreDist.length; j++) {
                        prevIterationTargetClassScoreDistribution[j] = prevIterScoreDist[j][targetClassIndex];
                    }
                    double batchTTestStatistic = batchTtest.pairedT(currentIterationTargetClassScoreDistribution, prevIterationTargetClassScoreDistribution);
                    batchPartitionIterationBackTtestScore.addValue(batchTTestStatistic);
                    batchIterationBackTtestScore.addValue(batchTTestStatistic);
                    //insert t-test to the attributes
                    AttributeInfo batchTTestStatisticAttr = new AttributeInfo
                            ("batchTtestScoreOnPartition_"+partitionIndex+"_iterationBack_"+numOfIterationsBack, Column.columnType.Numeric, batchTTestStatistic, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTTestStatisticAttr);
                }
                else{
                    AttributeInfo batchTTestStatisticAttr = new AttributeInfo
                            ("batchTtestScoreOnPartition_"+partitionIndex+"_iterationBack_"+numOfIterationsBack, Column.columnType.Numeric, -1, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTTestStatisticAttr);
                }
            }
            //max
            AttributeInfo batchPartitionTtestScoreMax = new AttributeInfo
                    ("batchPartition_" +partitionIndex+"_TtestScoreMax", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getMax(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMax);
            //min
            AttributeInfo batchPartitionTtestScoreMin = new AttributeInfo
                    ("batchPartition_" +partitionIndex+"_TtestScoreMin", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getMin(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMin);
            //mean
            AttributeInfo batchPartitionTtestScoreMean = new AttributeInfo
                    ("batchPartition_"+partitionIndex+"_TtestScoreMean", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getMean(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMean);
            //std
            AttributeInfo batchPartitionTtestScoreStd = new AttributeInfo
                    ("batchPartition_"+partitionIndex+"_TtestScoreStd", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getStandardDeviation(), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreStd);
            //p-50
            AttributeInfo batchPartitionTtestScoreMedian = new AttributeInfo
                    ("batchPartition_"+partitionIndex+"_TtestScoreMedian", Column.columnType.Numeric, batchPartitionIterationBackTtestScore.getPercentile(50), -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchPartitionTtestScoreMedian);
        }
        //statistics on T-test scores for all batch
        //max
        AttributeInfo batchTtestScoreMax = new AttributeInfo
                ("batchTtestScoreMax", Column.columnType.Numeric, batchIterationBackTtestScore.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMax);
        //min
        AttributeInfo batchTtestScoreMin = new AttributeInfo
                ("batchTtestScoreMin", Column.columnType.Numeric, batchIterationBackTtestScore.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMin);
        //mean
        AttributeInfo batchTtestScoreMean = new AttributeInfo
                ("batchTtestScoreMean", Column.columnType.Numeric, batchIterationBackTtestScore.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMean);
        //std
        AttributeInfo batchTtestScoreStd = new AttributeInfo
                ("batchTtestScoreStd", Column.columnType.Numeric, batchIterationBackTtestScore.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreStd);
        //p-50
        AttributeInfo batchTtestScoreMedian = new AttributeInfo
                ("batchTtestScoreMedian", Column.columnType.Numeric, batchIterationBackTtestScore.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), batchTtestScoreMedian);

        return instanceAttributesToReturn;
    }
}
