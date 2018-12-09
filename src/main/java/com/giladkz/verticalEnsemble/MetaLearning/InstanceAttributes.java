package com.giladkz.verticalEnsemble.MetaLearning;

import com.giladkz.verticalEnsemble.Data.*;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.counting;
import static java.util.function.Function.identity;

import java.util.*;

import static com.giladkz.verticalEnsemble.GeneralFunctions.EvaluationAnalysisFunctions.*;


public class InstanceAttributes {
    private List<Integer> numOfIterationsBackToAnalyze = Arrays.asList(1,3,5,10);

    public TreeMap<Integer, AttributeInfo> getInstanceAssignmentMetaFeatures
            (Dataset trainingDataset, Dataset testDataset, int currentIterationIndex,
             TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration,
             EvaluationPerIteraion unifiedDatasetEvaulationResults, int targetClassIndex,
             int instancePos, int assignedLabel, Properties properties) {

        TreeMap<Integer, AttributeInfo> instanceAttributesToReturn = new TreeMap<>();

        //insert label
        AttributeInfo assignedLabelAtt = new AttributeInfo
                ("assignedLabel", Column.columnType.Discrete, assignedLabel, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), assignedLabelAtt);

        //score by partition
        //HashMap<Integer, Double> partitionsInstanceEvaulatioInfos = new HashMap<>();
        //important: partitionIndex is for the 2 classifier or more?
        HashMap<Integer, double[][]> iterationScoreDistPerPartition = new HashMap<>();
        double instanceSumScore = 0;
        double instanceScoreDelta = 0;
        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            double instanceScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions()[instancePos][targetClassIndex];
            //partitionsInstanceEvaulatioInfos.put(partitionIndex, instanceScore);

            //insert score per partition
            AttributeInfo scoreClassifier = new AttributeInfo
                    ("scoreByClassifier" + partitionIndex, Column.columnType.Numeric, instanceScore, -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), scoreClassifier);

            //preparation for next calcs
            iterationScoreDistPerPartition.put(partitionIndex, evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions());
            instanceSumScore += instanceScore;
            if (partitionIndex%2==1){
                instanceScoreDelta += instanceScore;
            }
            else{
                instanceScoreDelta -= instanceScore;
            }
        }

        //insert the AVG score of the instance
        double instanceAvgScore = (instanceSumScore)/(evaluationResultsPerSetAndInteration.keySet().size());
        AttributeInfo instanceAvgAtt = new AttributeInfo
                ("instanceAVGScore", Column.columnType.Numeric, instanceAvgScore, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(),instanceAvgAtt);

        //insert the delta between the two classifiers
        double instanceClassifiersDeltaScore= Math.abs(instanceScoreDelta);
        AttributeInfo instanceClassifiersDeltaAtt = new AttributeInfo
                ("instanceClassifiersDeltaScore", Column.columnType.Numeric, instanceClassifiersDeltaScore, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaAtt);

        //percentile calculations
        double[] percentilePerClassPerInstancePerPartition = calculatePercentileClassificationResults(iterationScoreDistPerPartition, targetClassIndex, instancePos);
        double instanceDeltaPercentile = 0;
        for (int partitionIndex : iterationScoreDistPerPartition.keySet()){
           double instancePercentile = percentilePerClassPerInstancePerPartition[partitionIndex];
           //insert percentile per class
            AttributeInfo percentileByClassifier = new AttributeInfo
                    ("instancePercentileByClassifier" + partitionIndex, Column.columnType.Numeric, instancePercentile, -1);
            instanceAttributesToReturn.put(instanceAttributesToReturn.size(), percentileByClassifier);
            if (partitionIndex%2==1){
                instanceDeltaPercentile += instancePercentile;
            }
            else{
                instanceDeltaPercentile -= instancePercentile;
            }
        }
        //insert percentile delta
        double instanceClassifiersDeltaPercentile= Math.abs(instanceDeltaPercentile);
        AttributeInfo instanceClassifiersDeltaPercentileAtt = new AttributeInfo("instanceClassifiersDeltaScore", Column.columnType.Numeric, instanceClassifiersDeltaPercentile, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaPercentileAtt);


        //iterations back
        HashMap<Integer, DescriptiveStatistics> instanceScoreByPartitionAndIteration = new HashMap<>(); //partition --> scores of all iterations
        HashMap<Integer, DescriptiveStatistics> instanceDeltaScoreByPartitionAndIteration = new HashMap<>(); // partition-> delta scores from curr iter


        for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
            instanceScoreByPartitionAndIteration.put(partitionIndex, new DescriptiveStatistics());
            instanceDeltaScoreByPartitionAndIteration.put(partitionIndex, new DescriptiveStatistics());

            for (Integer numOfIterationsBack : numOfIterationsBackToAnalyze) {
                //need to cut it - if no iteration: put -1 (or '?') and not skip the iteration
                if (currentIterationIndex-numOfIterationsBack>=0){
                    double instanceCurrIterationScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions()[instancePos][targetClassIndex];
                    double prevInstanceScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex-numOfIterationsBack).getScoreDistributions()[instancePos][targetClassIndex];

                    instanceScoreByPartitionAndIteration.get(partitionIndex).addValue(prevInstanceScore);
                    //insert previous score
                    AttributeInfo instancePrevScore = new AttributeInfo
                            ("instancePrev" + numOfIterationsBack + "IterationsScoreClassifier" + partitionIndex, Column.columnType.Numeric, prevInstanceScore, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevScore);

                    //insert previous score - delta
                    double prevDeltaScore = Math.abs(instanceCurrIterationScore-prevInstanceScore);
                    instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).addValue(prevDeltaScore);
                    AttributeInfo instancePrevDeltaScore = new AttributeInfo
                            ("instancePrev" + numOfIterationsBack + "IterationsDeltaScoreClassifier" + partitionIndex, Column.columnType.Numeric, prevDeltaScore, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevDeltaScore);
                }else{
                    AttributeInfo instancePrevDeltaScore = new AttributeInfo
                            ("instancePrev" + numOfIterationsBack + "IterationsDeltaScoreClassifier" + partitionIndex, Column.columnType.Numeric, -1.0, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevDeltaScore);
                }
            }
//            if (currentIterationIndex-numOfIterationsBack>=0) {
                //stats on the iterations scores
                //max
                AttributeInfo instancePrevIterationMax = new AttributeInfo
                        ("instancePrevIterationMax" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getMax(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMax);
                //min
                AttributeInfo instancePrevIterationMin = new AttributeInfo
                        ("instancePrevIterationMin" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getMin(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMin);
                //mean
                AttributeInfo instancePrevIterationMean = new AttributeInfo
                        ("instancePrevIterationMean" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getMean(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMean);
                //std
                AttributeInfo instancePrevIterationStd = new AttributeInfo
                        ("instancePrevIterationStd" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getStandardDeviation(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationStd);
                //p-50
                AttributeInfo instancePrevIterationMedian = new AttributeInfo
                        ("instancePrevIterationMedian" + partitionIndex, Column.columnType.Numeric, instanceScoreByPartitionAndIteration.get(partitionIndex).getPercentile(50), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePrevIterationMedian);

                //stats on the iterations delta scores
                //max
                AttributeInfo instanceDeltaPrevIterationMax = new AttributeInfo
                        ("instanceDeltaPrevIterationMax" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getMax(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMax);
                //min
                AttributeInfo instanceDeltaPrevIterationMin = new AttributeInfo
                        ("instanceDeltaPrevIterationMin" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getMin(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMin);
                //mean
                AttributeInfo instanceDeltaPrevIterationMean = new AttributeInfo
                        ("instanceDeltaPrevIterationMean" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getMean(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMean);
                //std
                AttributeInfo instanceDeltaPrevIterationStd = new AttributeInfo
                        ("instanceDeltaPrevIterationStd" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getStandardDeviation(), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationStd);
                //p-50
                AttributeInfo instanceDeltaPrevIterationMedian = new AttributeInfo
                        ("instanceDeltaPrevIterationMedian" + partitionIndex, Column.columnType.Numeric, instanceDeltaScoreByPartitionAndIteration.get(partitionIndex).getPercentile(50), -1);
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevIterationMedian);
//            }
        }

        //collect stats on the inner delta per iteration
        DescriptiveStatistics innerIterationDelta = new DescriptiveStatistics();
        for (int iterationBackInd = 0; iterationBackInd < instanceDeltaScoreByPartitionAndIteration.get(0).getValues().length; iterationBackInd++) {
            innerIterationDelta.addValue(Math.abs(instanceDeltaScoreByPartitionAndIteration.get(0).getValues()[iterationBackInd] - instanceDeltaScoreByPartitionAndIteration.get(1).getValues()[iterationBackInd]));
        }
        //stats on the inner iterations delta scores
        //max
        AttributeInfo instanceDeltaPrevInnerIterationMax = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMax", Column.columnType.Numeric, innerIterationDelta.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMax);
        //min
        AttributeInfo instanceDeltaPrevInnerIterationMin = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMin", Column.columnType.Numeric, innerIterationDelta.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMin);
        //mean
        AttributeInfo instanceDeltaPrevInnerIterationMean = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMean", Column.columnType.Numeric, innerIterationDelta.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMean);
        //std
        AttributeInfo instanceDeltaPrevInnerIterationStd = new AttributeInfo
                ("instanceDeltaPrevInnerIterationStd", Column.columnType.Numeric, innerIterationDelta.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationStd);
        //p-50
        AttributeInfo instanceDeltaPrevInnerIterationMedian = new AttributeInfo
                ("instanceDeltaPrevInnerIterationMedian", Column.columnType.Numeric, innerIterationDelta.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceDeltaPrevInnerIterationMedian);

        //percentile on all previous iterations score
        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            HashMap<Integer, double[][]> prevIterationScoreDistPerPartition = new HashMap<>();
            if (currentIterationIndex >= numOfIterationsBack) {
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                    prevIterationScoreDistPerPartition.put(partitionIndex, evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex - numOfIterationsBack).getScoreDistributions());
                }
                double[] percentilePerClassPerInstancePerPartitionPrevIteration = calculatePercentileClassificationResults(prevIterationScoreDistPerPartition, targetClassIndex, instancePos);
                double instancePrevIterationDeltaPercentile = 0;
                for (int partitionIndex : prevIterationScoreDistPerPartition.keySet()) {
                    double instancePercentile = percentilePerClassPerInstancePerPartitionPrevIteration[partitionIndex];
                    //insert percentile per class
                    AttributeInfo percentilePrevIteratinByClassifier = new AttributeInfo
                            ("instancePrevIteration" + numOfIterationsBack + "PercentileByClassifier" + partitionIndex, Column.columnType.Numeric, instancePercentile, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), percentilePrevIteratinByClassifier);
                    if (partitionIndex % 2 == 1) {
                        instancePrevIterationDeltaPercentile += instancePercentile;
                    } else {
                        instancePrevIterationDeltaPercentile -= instancePercentile;
                    }
                }
                //insert percentile delta
                double instanceClassifiersDeltaPercentilePrevIteration = Math.abs(instancePrevIterationDeltaPercentile);
                AttributeInfo instanceClassifiersDeltaPercentileAttPrevIteration = new AttributeInfo("instanceClassifiersDeltaScore" + numOfIterationsBack, Column.columnType.Numeric, instanceClassifiersDeltaPercentilePrevIteration, testDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaPercentileAttPrevIteration);
            }
            //no iteration back - insert -1 in the value
            else {
                for (int partitionIndex : prevIterationScoreDistPerPartition.keySet()) {
                    AttributeInfo percentilePrevIteratinByClassifierNoIterationBack = new AttributeInfo
                            ("percentilePrevIteratinByClassifierNoIterationBack" + partitionIndex, Column.columnType.Numeric, -1.0, -1);
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), percentilePrevIteratinByClassifierNoIterationBack);
                }
                AttributeInfo instanceClassifiersDeltaPercentileAttNoIterationBack = new AttributeInfo("instanceClassifiersDeltaPercentileAttPrevIterationNoIterationBack", Column.columnType.Numeric, -1.0, testDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceClassifiersDeltaPercentileAttNoIterationBack);
            }
        }

        //data from the unified dataset - no deltas

        //score
        double instanceScoreUnifiedDataset = unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions()[instancePos][targetClassIndex];
        AttributeInfo instanceScoreUnifiedDatasetAtt = new AttributeInfo("instanceScoreUnifiedDataset", Column.columnType.Numeric, instanceScoreUnifiedDataset, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScoreUnifiedDatasetAtt);
        //percentile
        HashMap<Integer, double[][]> instancePercentileUnifiedDatasetMap = new HashMap<>();
        instancePercentileUnifiedDatasetMap.put(0,unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions());
        double instancePercentileUnifiedDataset = calculatePercentileClassificationResults(instancePercentileUnifiedDatasetMap, targetClassIndex, instancePos)[0];
        AttributeInfo instancePercentileUnifiedDatasetAtt = new AttributeInfo("instancePercentileUnifiedDataset", Column.columnType.Numeric, instancePercentileUnifiedDataset, testDataset.getNumOfClasses());
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentileUnifiedDatasetAtt);
        //iterations back
        DescriptiveStatistics scoresPerIterationUnifiedDataset = new DescriptiveStatistics();
        DescriptiveStatistics percentilePerIterationUnifiedDataset = new DescriptiveStatistics();
        for (int numOfIterationsBack : numOfIterationsBackToAnalyze) {
            if (currentIterationIndex >= numOfIterationsBack) {
                //score
                double scorePerIterationBack = unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex - numOfIterationsBack).getScoreDistributions()[instancePos][targetClassIndex];
                scoresPerIterationUnifiedDataset.addValue(scorePerIterationBack);
                AttributeInfo instanceScoreUnifiedDatasetPerIterationAtt = new AttributeInfo("instanceScoreUnifiedDatasetIteration" + numOfIterationsBack, Column.columnType.Numeric, scorePerIterationBack, testDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScoreUnifiedDatasetPerIterationAtt);

                //percentile
                HashMap<Integer, double[][]> instancePercentileUnifiedDatasetMapPerIteration = new HashMap<>();
                instancePercentileUnifiedDatasetMapPerIteration.put(0,unifiedDatasetEvaulationResults.getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions());
                percentilePerIterationUnifiedDataset.addValue(calculatePercentileClassificationResults(instancePercentileUnifiedDatasetMapPerIteration, targetClassIndex, instancePos)[0]);
            }
            else{
                AttributeInfo instanceScoreUnifiedDatasetPerIterationAttNoIterationBack = new AttributeInfo("instanceScoreUnifiedDatasetIterationNoIterationBack", Column.columnType.Numeric, -1.0, testDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScoreUnifiedDatasetPerIterationAttNoIterationBack);
                //how to insert -1 to the percentile calculation??
            }
        }

        //insert statistics on the previous iterations for the unified dataset
        //max
        AttributeInfo instanceScorePrevInnerIterationMaxUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMaxUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMaxUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMaxUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMaxUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getMax(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMaxUnifiedDataset);
        //min
        AttributeInfo instanceScorePrevInnerIterationMinUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMinUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMinUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMinUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMinUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getMin(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMinUnifiedDataset);
        //mean
        AttributeInfo instanceScorePrevInnerIterationMeanUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMeanUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMeanUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMeanUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMeanUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getMean(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMeanUnifiedDataset);

        //std
        AttributeInfo instanceScorePrevInnerIterationStdUnifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationStdUnifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationStdUnifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationStdUnifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationStdUnifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getStandardDeviation(), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationStdUnifiedDataset);
        //p-50
        AttributeInfo instanceScorePrevInnerIterationMedianunifiedDataset = new AttributeInfo
                ("instanceScorePrevInnerIterationMedianunifiedDataset", Column.columnType.Numeric, scoresPerIterationUnifiedDataset.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceScorePrevInnerIterationMedianunifiedDataset);
        AttributeInfo instancePercentilePrevInnerIterationMedianunifiedDataset = new AttributeInfo
                ("instancePercentilePrevInnerIterationMedianunifiedDataset", Column.columnType.Numeric, percentilePerIterationUnifiedDataset.getPercentile(50), -1);
        instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentilePrevInnerIterationMedianunifiedDataset);

        //column data as in the dataset
        //need to understand how to get the instance data from the data set
        List<ColumnInfo> datasetColInfo = trainingDataset.getAllColumns(false);
//        List<ColumnInfo> datasetColInfo = trainingDataset.getColumns(Arrays.asList()); //need to be the col indices and not instance index
        for (ColumnInfo colInf: datasetColInfo) {
            Column col = colInf.getColumn();

            if(col.getType() == Column.columnType.Numeric){
                double[] columnData = (double[])(col.getValues());
                double instancePosData = (double)(col.getValue(instancePos));

                double maxColValue = Arrays.stream(columnData).max().getAsDouble();
                double minColValue = Arrays.stream(columnData).min().getAsDouble();
                double[] normalizedColumnData = norm100and0(maxColValue, minColValue, columnData);
                double normalizedInstancePosData = ((instancePosData - minColValue + 0.01)/(maxColValue - minColValue + 0.01))*100;

                //percentile for the instance from total column
                Percentile p = new Percentile();
                p.setData(normalizedColumnData);
                double instancePercentileColumn = p.evaluate(normalizedInstancePosData);
                AttributeInfo instancePercentileColumnAttr = new AttributeInfo
                        ("instancePercentileColumn_" + instancePos + "_" + currentIterationIndex, Column.columnType.Numeric, instancePercentileColumn, testDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instancePercentileColumnAttr);


                //percentile for the instance from total column - assign class and
                //percentile for the instance from total column - higher conf. level
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                    double[][] allColumnsScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
                    ArrayList<Double> assignedLabelIndeciesTemp = new ArrayList<>();
                    ArrayList<Double> higherValueIndeciesTemp = new ArrayList<>();
                    for (int i = 0; i < allColumnsScore.length; i++) {
                        if (allColumnsScore[i][assignedLabel] > 0.5){
                            assignedLabelIndeciesTemp.add((double)(col.getValue(i)));
                        }
                        if(allColumnsScore[i][assignedLabel] > allColumnsScore[instancePos][assignedLabel]){
                            higherValueIndeciesTemp.add((double)(col.getValue(i)));
                        }
                    }
                    double[] assignedLabelIndecies = new double[assignedLabelIndeciesTemp.size()];
                    for (int i = 0; i < assignedLabelIndeciesTemp.size(); i++) {
                        assignedLabelIndecies[i] = assignedLabelIndeciesTemp.get(i);
                    }
                    double[] normAssignedLabelIndecies = norm100and0(Arrays.stream(assignedLabelIndecies).max().getAsDouble(), Arrays.stream(assignedLabelIndecies).min().getAsDouble(), assignedLabelIndecies);

                    p.setData(normAssignedLabelIndecies);
                    double assignedLabelPercentile = p.evaluate(normalizedInstancePosData);
                    AttributeInfo assignedLabelPercentileAttr = new AttributeInfo
                            ("assignedLabelPercentile_" + instancePos + "_" + currentIterationIndex, Column.columnType.Numeric, assignedLabelPercentile, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), assignedLabelPercentileAttr);

                    double[] higherValueIndecies = new double[higherValueIndeciesTemp.size()];
                    for (int i = 0; i < higherValueIndeciesTemp.size(); i++) {
                        higherValueIndecies[i] = higherValueIndeciesTemp.get(i);
                    }
                    double[] normHigherValueIndecies = norm100and0(Arrays.stream(higherValueIndecies).max().getAsDouble(), Arrays.stream(higherValueIndecies).min().getAsDouble(), higherValueIndecies);
                    p.setData(normHigherValueIndecies);
                    double higherValuePercentile = p.evaluate(normAssignedLabelIndecies);
                    AttributeInfo higherValuePercentileAttr = new AttributeInfo
                            ("higherValuePercentile_" + instancePos + "_" + currentIterationIndex, Column.columnType.Numeric, higherValuePercentile, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), higherValuePercentileAttr );
                }

            }
            else if (col.getType() == Column.columnType.Discrete){
                //priori
                int[] columnData = (int[])(col.getValues());
                int instancePosData = (int)(col.getValue(instancePos));

                //mode function
                HashMap<Integer, Double> valuesCount = new HashMap<>();
                double totalValues = 0.0;
                for (int value : columnData){
                    if (valuesCount.containsKey(value)) {
                        valuesCount.put(value, valuesCount.get(value)+1.0);
                        totalValues++;
                    } else {
                        valuesCount.put(value,1.0);
                        totalValues++;
                    }
                }
                //prob to the value in the dataset
                double instanceValueCount = valuesCount.get(instancePosData);
                double instanceValueProb = instanceValueCount/totalValues;
                AttributeInfo instanceValueProbAttr = new AttributeInfo
                        ("instanceValueProb_" + instancePos + "_" + currentIterationIndex, Column.columnType.Numeric, instanceValueProb, testDataset.getNumOfClasses());
                instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceValueProbAttr);
                //Statistics on the probability of each value given all other values (in the assigned class)
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()) {
                    double[][] allColumnsScore = evaluationResultsPerSetAndInteration.get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex).getScoreDistributions();
                    HashMap<Integer, Double> valuesCountLabeled = new HashMap<>();
                    HashMap<Integer, Double> valuesCountHigher = new HashMap<>();
                    double valueCountLabeledTotal = 0.0;
                    double valueCountHigherTotal = 0.0;
                    for (int i = 0; i < allColumnsScore.length; i++) {
                        int value = (int)col.getValue(i);
                        if (allColumnsScore[i][assignedLabel] > 0.5){
                            valueCountLabeledTotal++;
                            if(valuesCountLabeled.containsKey(value)){
                                valuesCountLabeled.put(value, valuesCountLabeled.get(value)+1.0);
                            }
                            else{
                                valuesCountLabeled.put(value,1.0);
                            }
                        }
                        if(allColumnsScore[i][assignedLabel] > allColumnsScore[instancePos][assignedLabel]){
                            valueCountHigherTotal++;
                            if(valuesCountHigher.containsKey(value)){
                                valuesCountHigher.put(value, valuesCountHigher.get(value)+1.0);
                            }
                            else{
                                valuesCountHigher.put(value,1.0);
                            }
                        }
                    }
                    double instanceValueLabeledCount = valuesCountLabeled.get(instancePosData);
                    double instanceValueLabeledProb = instanceValueLabeledCount/valueCountLabeledTotal;
                    double instanceValueHigherCount = valuesCountHigher.get(instancePosData);
                    double instanceValueHigherProb = instanceValueHigherCount/valueCountHigherTotal;

                    AttributeInfo instanceValueLabeledProbAttr = new AttributeInfo
                            ("instanceValueLabeledProb" + instancePos + "_" + currentIterationIndex + '_' + partitionIndex, Column.columnType.Numeric, instanceValueLabeledProb, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceValueLabeledProbAttr);
                    AttributeInfo instanceValueHigherProbAttr = new AttributeInfo
                            ("instanceValueHigherProb" + instancePos + "_" + currentIterationIndex + '_' + partitionIndex, Column.columnType.Numeric, instanceValueHigherProb, testDataset.getNumOfClasses());
                    instanceAttributesToReturn.put(instanceAttributesToReturn.size(), instanceValueHigherProbAttr);
                }
            }
            //else continue;
        }

        //fix NaN: convert to -1.0
        for (Map.Entry<Integer,AttributeInfo> entry : instanceAttributesToReturn.entrySet()){
            AttributeInfo ai = entry.getValue();
            if (ai.getAttributeType() == Column.columnType.Numeric){
                double aiVal = (double) ai.getValue();
                if (Double.isNaN(aiVal)){
                    ai.setValue(-1.0);
                }
            }
        }
        return instanceAttributesToReturn;
    }

    private double[] norm100and0 (double max, double min, double[] arr){
        double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = ((arr[i] - min + 0.01)/(max - min + 0.01))*100;
        }
        return result;
    }
}
