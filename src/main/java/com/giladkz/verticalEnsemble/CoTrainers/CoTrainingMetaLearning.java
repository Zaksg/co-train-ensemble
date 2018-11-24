package com.giladkz.verticalEnsemble.CoTrainers;

import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.MetaLearning.InstanceAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.ScoreDistributionBasedAttributes;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class CoTrainingMetaLearning extends CoTrainerAbstract {
    private Properties properties;

    @Override
    public Dataset Train_Classifiers(HashMap<Integer, List<Integer>> feature_sets, Dataset dataset, int initial_number_of_labled_samples,
                                     int num_of_iterations, HashMap<Integer, Integer> instances_per_class_per_iteration, String original_arff_file,
                                     int initial_unlabeled_set_size, double weight, DiscretizerAbstract discretizer, int exp_id, String arff,
                                     int iteration, double weight_for_log, boolean use_active_learning, int random_seed) throws Exception {

        /*This set is meta features analyzes the scores assigned to the unlabeled training set at each iteration.
        * Its possible uses include:
        * a) Providing a stopping criteria (i.e. need to go one iteration (or more) back because we're going off-course
        * b) Assisting in the selection of unlabeled samples to be added to the labeled set*/
        ScoreDistributionBasedAttributes scoreDistributionBasedAttributes = new ScoreDistributionBasedAttributes();

        properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        /* We start by partitioning the dataset based on the sets of features this function receives as a parameter */
        HashMap<Integer,Dataset> datasetPartitions = new HashMap<>();
        for (int index : feature_sets.keySet()) {
            Dataset partition = dataset.replicateDatasetByColumnIndices(feature_sets.get(index));

            datasetPartitions.put(index, partition);
        }

        /* Randomly select the labeled instances from the training set. The remaining ones will be used as the unlabeled.
         * It is important that we use a fixed random seed for repeatability */
        List<Integer> labeledTrainingSetIndices = getLabeledTrainingInstancesIndices(dataset,initial_number_of_labled_samples,true,random_seed);

        /* If the unlabeled training set is larger than the specified parameter, we will sample X instances to
         * serve as the pool. TODO: replenish the pool upon sampling (although given the sizes it's not such a big deal */
        List<Integer> unlabeledTrainingSetIndices = new ArrayList<>();
        Fold trainingFold = dataset.getTrainingFolds().get(0); //There should only be one training fold in this type of project
        if (trainingFold.getIndices().size()-initial_number_of_labled_samples > initial_unlabeled_set_size) {
            //ToDo: add a random sampling function
        }
        else {
            for (int index : trainingFold.getIndices()) {
                if (!labeledTrainingSetIndices.contains(index)) {
                    unlabeledTrainingSetIndices.add(index);
                }
            }
        }

        //before we begin the co-training process, we test the performance on the original dataset
        RunExperimentsOnTestSet(exp_id, iteration, -1, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);

        //And now we can begin the iterative process

        //this object saves the results for the partitioned dataset. It is of the form parition -> iteration index -> results
        HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration = new HashMap<>();

        //this object save the results of the runs of the unified datasets (original labeled + labeled during the co-training process).
        EvaluationPerIteraion unifiedDatasetEvaulationResults = new EvaluationPerIteraion();


        for (int i=0; i<num_of_iterations; i++) {
            /*for each set of features, train a classifier on the labeled training set and: a) apply it on the
            unlabeled set to select the samples that will be added; b) apply the new model on the test set, so that
            we can know during the analysis how we would have done on the test set had we stopped in this particular iteration*/


            //step 1 - train the classifiers on the labeled training set and run on the unlabeled training set
            System.out.println("labaled: " + labeledTrainingSetIndices.size() + ";  unlabeled: " + unlabeledTrainingSetIndices.size() );

            for (int partitionIndex : feature_sets.keySet()) {
                EvaluationInfo evaluationResults = runClassifier(properties.getProperty("classifier"),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                        datasetPartitions.get(partitionIndex).generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices), properties);

                if (!evaluationResultsPerSetAndInteration.containsKey(partitionIndex)) {
                    evaluationResultsPerSetAndInteration.put(partitionIndex, new EvaluationPerIteraion());
                }
                evaluationResultsPerSetAndInteration.get(partitionIndex).addEvaluationInfo(evaluationResults, i);
            }

            //now we run the classifier trained on the unified set
            EvaluationInfo unifiedSetEvaluationResults = runClassifier(properties.getProperty("classifier"),
                    dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices),
                    dataset.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices), properties);
            unifiedDatasetEvaulationResults.addEvaluationInfo(unifiedSetEvaluationResults, i);

            //step 2 - get the indices of the items we want to label (separately for each class)
            HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass = new HashMap<>();
            HashMap<Integer, List<Integer>> instancesPerPartition = new HashMap<>();
            //these are the indices of the array provided to Weka. They need to be converted to the Dataset indices
            GetIndicesOfInstancesToLabelBasic(dataset, instances_per_class_per_iteration, evaluationResultsPerSetAndInteration, instancesToAddPerClass, random_seed, unlabeledTrainingSetIndices, instancesPerPartition);

            super.WriteInformationOnAddedItems(instancesToAddPerClass, i, exp_id,iteration,weight_for_log,instancesPerPartition, properties, dataset);

            //step 3 - set the class labels of the newly labeled instances to what we THINK they are
            for (int classIndex : instancesToAddPerClass.keySet()) {
                //because the columns of the partitions are actually the columns of the original dataset, there is no problem changing things only there
                dataset.updateInstanceTargetClassValue(new ArrayList<>(instancesToAddPerClass.get(classIndex).keySet()), classIndex);
            }


            //step 4 - add the selected instances to the labeled training set and remove them from the unlabeled set
            //IMPORTANT: when adding the unlabeled samples, it must be with the label I ASSUME they possess.
            List<Integer> allIndeicesToAdd = new ArrayList<>();
            for (int classIndex : instancesToAddPerClass.keySet()) {
                allIndeicesToAdd.addAll(new ArrayList<Integer>(instancesToAddPerClass.get(classIndex).keySet()));
            }
            labeledTrainingSetIndices.addAll(allIndeicesToAdd);
            unlabeledTrainingSetIndices = unlabeledTrainingSetIndices.stream().filter(line -> !allIndeicesToAdd.contains(line)).collect(Collectors.toList());

            //step 5 - train the models using the current instances and apply them to the test set
            RunExperimentsOnTestSet(exp_id, iteration, i, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions, labeledTrainingSetIndices, properties);



            //step 6 - generate the meta features
            generateMetaFeatures(dataset, labeledTrainingSetIndices, unlabeledTrainingSetIndices, evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, i, properties);

            //HashMap<Integer,AttributeInfo> scoreDistributionMetaFeatures = scoreDistributionBasedAttributes.getScoreDistributionBasedAttributes()
        }
        return null;
    }


    @Override
    public void Previous_Iterations_Analysis(EvaluationPerIteraion models, Dataset training_set_data, Dataset validation_set_data, int current_iteration) {

    }

    @Override
    public String toString() {
        return "CoTrainerMetaLearning";
    }


    /**
     * Used to generate the meta-features that will be used to guide the co-training process.
     * @param dataset The COMPLETE dataset (even the test set, so there's a need to be careful)
     * @param labeledTrainingSetIndices The indices of all the labeled training instances
     * @param unlabeledTrainingSetIndices The indices of all the unlabeled training instances
     * @param evaluationResultsPerSetAndInteration The evaluation results of EACH PARTITION across different indices
     * @param unifiedDatasetEvaulationResults The evaluation results of the UNIFIED (i.e. all features) features set across differnet iterations
     * @param currentIterationIndex The index of the current iteration
     * @param properties Configuration values
     * @return
     * @throws Exception
     */
    private HashMap<Integer,AttributeInfo> generateMetaFeatures(
            Dataset dataset, List<Integer> labeledTrainingSetIndices
            , List<Integer> unlabeledTrainingSetIndices
            , HashMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInteration
            , EvaluationPerIteraion unifiedDatasetEvaulationResults
            , int currentIterationIndex, Properties properties) throws Exception{
        Loader loader = new Loader();
        String tempFilePath = properties.getProperty("tempDirectory") + "temp.arff";
        Files.deleteIfExists(Paths.get(tempFilePath));
        FoldsInfo foldsInfo = new FoldsInfo(1,0,0,1,-1,0,0,0,-1,true, FoldsInfo.foldType.Train);

        //generate the labeled training instances dataset
        ArffSaver s= new ArffSaver();
        s.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,labeledTrainingSetIndices));
        s.setFile(new File(tempFilePath));
        s.writeBatch();
        BufferedReader reader = new BufferedReader(new FileReader(tempFilePath));
        Dataset labeledTrainingDataset = loader.readArff(reader, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo);
        reader.close();

        File file = new File(tempFilePath);

        if(!file.delete())
        {
            throw new Exception("Temp file not deleted1");
        }

        //generate the unlabeled training instances dataset
        s= new ArffSaver();
        s.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,unlabeledTrainingSetIndices));
        s.setFile(new File(tempFilePath));
        s.writeBatch();
        BufferedReader reader1 = new BufferedReader(new FileReader(tempFilePath));
        Dataset unlabeledTrainingDataset = loader.readArff(reader1, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo);
        reader1.close();

        file = new File(tempFilePath);

        if(!file.delete())
        {
            throw new Exception("Temp file not deleted2");
        }

        //evaluationResultsPerSetAndInteration
        ScoreDistributionBasedAttributes scoreDistributionBasedAttributes = new ScoreDistributionBasedAttributes();
        InstanceAttributes instanceAttributes = new InstanceAttributes();

        int targetClassIndex = dataset.getMinorityClassIndex();

        //scoreDistributionBasedAttributes.getScoreDistributionBasedAttributes(labeledTrainingDataset,unlabeledTrainingDataset, currentIterationIndex, evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, dataset.getTargetColumnIndex(), properties);

//        for (int instancePos: unlabeledTrainingDataset.getIndices()) {
//            instanceAttributes.getInstanceAssignmentMetaFeatures(labeledTrainingDataset, unlabeledTrainingDataset, currentIterationIndex, evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, targetClassIndex, instancePos, unlabeledTrainingDataset.getInstancesClassByIndex(Arrays.asList(instancePos)).get(0), properties);
//        }
        return null;

    }
}

