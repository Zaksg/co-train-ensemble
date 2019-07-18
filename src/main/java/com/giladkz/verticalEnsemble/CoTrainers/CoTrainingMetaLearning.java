package com.giladkz.verticalEnsemble.CoTrainers;

import com.giladkz.verticalEnsemble.Data.*;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.MetaLearning.InstanceAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.InstancesBatchAttributes;
import com.giladkz.verticalEnsemble.MetaLearning.ScoreDistributionBasedAttributes;
import com.giladkz.verticalEnsemble.StatisticsCalculations.AUC;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Array;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.time.LocalDateTime;
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
        InstanceAttributes instanceAttributes = new InstanceAttributes();
        InstancesBatchAttributes instancesBatchAttributes = new InstancesBatchAttributes();

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
            for (int index : trainingFold.getIndices()) {
                if (!labeledTrainingSetIndices.contains(index) && unlabeledTrainingSetIndices.size() < initial_unlabeled_set_size
                        && new Random().nextInt(100)< 96) {
                    unlabeledTrainingSetIndices.add(index);
                }
            }
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
        //write meta data information in groups and not one by one
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeInstanceMetaDataInGroup = new HashMap<>();
        HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeBatchMetaDataInGroup = new HashMap<>();
        HashMap<ArrayList<Integer>,int[]> writeInsertBatchInGroup = new HashMap<>();
        HashMap<int[], Double> writeSampleBatchScoreInGroup = new HashMap<>();
        int writeCounterBin = 1;

        for (int i=0; i<num_of_iterations; i++) {
            /*for each set of features, train a classifier on the labeled training set and: a) apply it on the
            unlabeled set to select the samples that will be added; b) apply the new model on the test set, so that
            we can know during the analysis how we would have done on the test set had we stopped in this particular iteration*/


            //step 1 - train the classifiers on the labeled training set and run on the unlabeled training set
//            System.out.println("start iteration with: labaled: " + labeledTrainingSetIndices.size() + ";  unlabeled: " + unlabeledTrainingSetIndices.size() );

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

            /*enter the meta features generation here*/
            Dataset labeledToMetaFeatures = dataset;
            Dataset unlabeledToMetaFeatures = dataset;
            int targetClassIndex = dataset.getMinorityClassIndex();
            boolean getDatasetInstancesSucc = false;
            for (int numberOfTries = 0; numberOfTries < 5 && !getDatasetInstancesSucc; numberOfTries++) {
                try{
                    labeledToMetaFeatures = getDataSetByInstancesIndices(dataset,labeledTrainingSetIndices,properties);
                    unlabeledToMetaFeatures = getDataSetByInstancesIndices(dataset,unlabeledTrainingSetIndices,properties);
                    getDatasetInstancesSucc = true;
                }catch (Exception e){
                    Thread.sleep(1000);
//                    System.out.println("failed reading file, sleep for 1 second, for try: " + num_of_iterations + " at time: " + LocalDateTime.now());
                    getDatasetInstancesSucc = false;
                }
            }

            TreeMap<Integer, EvaluationPerIteraion> evaluationResultsPerSetAndInterationTree = new TreeMap<>(evaluationResultsPerSetAndInteration);

            //score distribution
            TreeMap<Integer,AttributeInfo> scoreDistributionCurrentIteration = new TreeMap<>();
            scoreDistributionCurrentIteration = scoreDistributionBasedAttributes.getScoreDistributionBasedAttributes(
                    unlabeledToMetaFeatures,labeledToMetaFeatures,
                    i, evaluationResultsPerSetAndInterationTree, unifiedDatasetEvaulationResults,
                    targetClassIndex/*dataset.getTargetColumnIndex()*/, properties);
            writeResultsToScoreDistribution(scoreDistributionCurrentIteration, i, exp_id, iteration, properties, dataset);

            /*pick 1000 batches, train on cloned dataset, and get the AUC*/
            ArrayList<ArrayList<Integer>> batchesInstancesList = new ArrayList<>();
            List<TreeMap<Integer,AttributeInfo>> instanceAttributeCurrentIterationList = new ArrayList<>();


            //pick random 1000 batches of 8 instances and get meta features
            //TO-DO: extract random selection to different class in order to control the changes of other selection methods
            Random rnd = new Random((i + Integer.parseInt(properties.getProperty("randomSeed"))));
            int numOfBatches = (int) Math.min(Integer.parseInt(properties.getProperty("numOfBatchedPerIteration")),Math.round(0.3*unlabeledTrainingSetIndices.size()));
            for (int batchIndex = 0; batchIndex < numOfBatches; batchIndex++) {
                ArrayList<Integer> instancesBatchOrginalPos = new ArrayList<>();
                ArrayList<Integer> instancesBatchSelectedPos = new ArrayList<>();

                HashMap<Integer, Integer> assignedLabelsOriginalIndex = new HashMap<>();
                HashMap<Integer, Integer> assignedLabelsSelectedIndex = new HashMap<>();
                HashMap<TreeMap<Integer,AttributeInfo>, int[]> writeInstanceMetaDataInGroupTemp = new HashMap<>();
                int class0counter = 0;
                int class1counter = 0;
                for (int partitionIndex : evaluationResultsPerSetAndInteration.keySet()){
                    //we need 8 distinct instances
                    for (int sampleIndex = 0; sampleIndex < Integer.parseInt(properties.getProperty("instancesPerBatch"))/2; sampleIndex++) {
                        int relativeIndex = rnd.nextInt(unlabeledTrainingSetIndices.size());

                        while(assignedLabelsSelectedIndex.containsKey(relativeIndex)){
                            relativeIndex = rnd.nextInt(unlabeledTrainingSetIndices.size());
                        }
                        Integer instancePos = unlabeledTrainingSetIndices.get(relativeIndex);
                        //add instance pos to batch list
                        instancesBatchOrginalPos.add(instancePos);
                        instancesBatchSelectedPos.add(relativeIndex);

                        //calculate instance class

                        int assignedClass;
                        double scoreClass0 = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions()[relativeIndex][0];
                        double scoreClass1 = evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions()[relativeIndex][1];
                        if (scoreClass0 > scoreClass1){
                            assignedClass = 0;
                            class0counter++;
                        }
                        else{
                            assignedClass = 1;
                            class1counter++;
                        }
                        assignedLabelsOriginalIndex.put(instancePos, assignedClass);
                        assignedLabelsSelectedIndex.put(relativeIndex, assignedClass);
                        //get instance meta features

                        TreeMap<Integer,AttributeInfo> instanceAttributeCurrentIteration = instanceAttributes.getInstanceAssignmentMetaFeatures(
                                unlabeledToMetaFeatures,dataset,
                                i, evaluationResultsPerSetAndInterationTree,
                                unifiedDatasetEvaulationResults, targetClassIndex,
                                relativeIndex,instancePos, assignedClass, properties);
                        instanceAttributeCurrentIterationList.add(instanceAttributeCurrentIteration);
                        int[] instanceInfoToWrite = new int[5];
                        instanceInfoToWrite[0]=exp_id;
                        instanceInfoToWrite[1]=iteration;
                        instanceInfoToWrite[2]=i;
                        instanceInfoToWrite[3]=instancePos;
                        instanceInfoToWrite[4]=batchIndex;
                        writeInstanceMetaDataInGroupTemp.put(instanceAttributeCurrentIteration, instanceInfoToWrite);
                        //writeResultsToInstanceMetaFeatures(instanceAttributeCurrentIteration, exp_id, iteration, i, instancePos, batchIndex, properties, dataset);
                    }
                }
                if (class0counter > Integer.parseInt(properties.getProperty("minNumberOfInstancesPerClassInAbatch"))
                        && class1counter > Integer.parseInt(properties.getProperty("minNumberOfInstancesPerClassInAbatch"))){
                    writeInstanceMetaDataInGroup.putAll(writeInstanceMetaDataInGroupTemp);
                    batchesInstancesList.add(instancesBatchOrginalPos);
                    TreeMap<Integer,AttributeInfo> batchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                            unlabeledToMetaFeatures,labeledToMetaFeatures,
                            i, evaluationResultsPerSetAndInterationTree,
                            unifiedDatasetEvaulationResults, targetClassIndex/*dataset.getTargetColumnIndex()*/,
                            instancesBatchSelectedPos, assignedLabelsSelectedIndex, properties);

                    int[] batchInfoToWrite = new int[3];
                    batchInfoToWrite[0]=exp_id;
                    batchInfoToWrite[1]=i;
                    batchInfoToWrite[2]=batchIndex;
                    writeBatchMetaDataInGroup.put(batchAttributeCurrentIterationList, batchInfoToWrite);
                    //writeResultsToBatchesMetaFeatures(batchAttributeCurrentIterationList, exp_id, i, batchIndex, properties, dataset);

                    //run the classifier with this batch: on cloned dataset and re-create the run-experiment method (Batches_Score)
                    writeSampleBatchScoreInGroup.putAll(runClassifierOnSampledBatch(exp_id, iteration, i, batchIndex, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions,assignedLabelsOriginalIndex, labeledTrainingSetIndices, properties));
//                    runClassifierOnSampledBatch(exp_id, iteration, i, batchIndex, dataset, dataset.getTestFolds().get(0), dataset.getTrainingFolds().get(0), datasetPartitions,assignedLabelsOriginalIndex, labeledTrainingSetIndices, properties);
//                    System.out.println("done sample random batches for batch: " + batchIndex + ", iteration: " + i);
                }
                writeInstanceMetaDataInGroupTemp.clear();
            }



            //step 2 - get the indices of the items we want to label (separately for each class)
            HashMap<Integer,HashMap<Integer,Double>> instancesToAddPerClass = new HashMap<>();
            HashMap<Integer, List<Integer>> instancesPerPartition = new HashMap<>();
            HashMap<Integer, Integer> selectedInstancesRelativeIndexes = new HashMap<>(); //index (relative) -> assigned class
            ArrayList<Integer> indicesOfAddedInstances = new ArrayList<>(); //index(original)
            //these are the indices of the array provided to Weka. They need to be converted to the Dataset indices
            GetIndicesOfInstancesToLabelBasicRelativeIndex(dataset, instances_per_class_per_iteration, evaluationResultsPerSetAndInteration, instancesToAddPerClass, random_seed, unlabeledTrainingSetIndices, instancesPerPartition, selectedInstancesRelativeIndexes, indicesOfAddedInstances);

            super.WriteInformationOnAddedItems(instancesToAddPerClass, i, exp_id,iteration,weight_for_log,instancesPerPartition, properties, dataset);

            //selected batch meta-data
            for (Integer instance: selectedInstancesRelativeIndexes.keySet()){
                Integer originalInstancePos = unlabeledTrainingSetIndices.get(instance);
                Integer assignedClass = selectedInstancesRelativeIndexes.get(instance);
                TreeMap<Integer,AttributeInfo> instanceAttributeCurrentIteration = instanceAttributes.getInstanceAssignmentMetaFeatures(
                        unlabeledToMetaFeatures,dataset,
                        i, evaluationResultsPerSetAndInterationTree,
                        unifiedDatasetEvaulationResults, targetClassIndex/*dataset.getTargetColumnIndex()*/,
                        instance,originalInstancePos, assignedClass, properties);
//                instanceAttributeCurrentIterationList.add(instanceAttributeCurrentIteration);
                int[] instanceInfoToWrite = new int[5];
                instanceInfoToWrite[0]=exp_id;
                instanceInfoToWrite[1]=iteration;
                instanceInfoToWrite[2]=i;
                instanceInfoToWrite[3]=originalInstancePos;
                instanceInfoToWrite[4]= -1;
                writeInstanceMetaDataInGroup.put(instanceAttributeCurrentIteration, instanceInfoToWrite);
//                writeResultsToInstanceMetaFeatures(instanceAttributeCurrentIteration, exp_id, iteration, i, originalInstancePos, -1, properties, dataset);
            }

            TreeMap<Integer,AttributeInfo>  selectedBatchAttributeCurrentIterationList = instancesBatchAttributes.getInstancesBatchAssignmentMetaFeatures(
                    unlabeledToMetaFeatures,labeledToMetaFeatures,
                    i, evaluationResultsPerSetAndInterationTree,
                    unifiedDatasetEvaulationResults, targetClassIndex/*dataset.getTargetColumnIndex()*/,
                    new ArrayList<>(selectedInstancesRelativeIndexes.keySet()), selectedInstancesRelativeIndexes, properties);
            int[] batchInfoToWrite = new int[3];
            batchInfoToWrite[0]=exp_id;
            batchInfoToWrite[1]=i;
            batchInfoToWrite[2]= -1 + iteration*(-1);
            writeBatchMetaDataInGroup.put(selectedBatchAttributeCurrentIterationList, batchInfoToWrite);
            //writeResultsToBatchesMetaFeatures(selectedBatchAttributeCurrentIterationList, exp_id, i,  -1 + iteration*(-1), properties, dataset);

            int[] insertBatchInfoToWrite = new int[3];
            insertBatchInfoToWrite[0]=exp_id;
            insertBatchInfoToWrite[1]=iteration;
            insertBatchInfoToWrite[2]=i;
            writeInsertBatchInGroup.put(indicesOfAddedInstances, insertBatchInfoToWrite);
            //writeToInstancesInBatchTbl(iteration*(-1), exp_id, iteration, indicesOfAddedInstances, properties);


            //write meta-data in groups of 20% of iteration
            if (i == (writeCounterBin*(num_of_iterations/5))-1){
                writeResultsToInstanceMetaFeaturesGroup(writeInstanceMetaDataInGroup, properties, dataset);
                writeResultsToBatchMetaFeaturesGroup(writeBatchMetaDataInGroup, properties, dataset);
                writeToInsertInstancesToBatchGroup(writeInsertBatchInGroup, properties);
                writeToBatchScoreTblGroup(writeSampleBatchScoreInGroup, properties);
                writeInstanceMetaDataInGroup.clear();
                writeBatchMetaDataInGroup.clear();
                writeInsertBatchInGroup.clear();
                writeSampleBatchScoreInGroup.clear();
                writeCounterBin++;
            }

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

            System.out.println("dataset: "+dataset.getName()+" done insert batch and run the classifier for iteration: " + i);


            /* old version call
            step 6 - generate the meta features - not relevant!!
            generateMetaFeatures(dataset, labeledTrainingSetIndices, unlabeledTrainingSetIndices, evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, i, properties);
            HashMap<Integer,AttributeInfo> scoreDistributionMetaFeatures = scoreDistributionBasedAttributes.getScoreDistributionBasedAttribute()            */
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

    private Dataset getDataSetByInstancesIndices (Dataset dataset, List<Integer> setIndices, Properties properties)throws Exception{

        Loader loader = new Loader();
        String tempFilePath = properties.getProperty("tempDirectory") + "temp.arff";
        Files.deleteIfExists(Paths.get(tempFilePath));
        FoldsInfo foldsInfo = new FoldsInfo(1,0,0,1,-1,0,0,0,-1,true, FoldsInfo.foldType.Train);

        //generate the labeled training instances dataset
        ArffSaver s= new ArffSaver();
        s.setInstances(dataset.generateSet(FoldsInfo.foldType.Train,setIndices));
        s.setFile(new File(tempFilePath));
        s.writeBatch();
        BufferedReader reader = new BufferedReader(new FileReader(tempFilePath));
        Dataset newDataset = loader.readArff(reader, 0, null, dataset.getTargetColumnIndex(), 1, foldsInfo);
        reader.close();

        File file = new File(tempFilePath);

        if(!file.delete())
        {
            throw new Exception("Temp file not deleted1");
        }
        return newDataset;
    }

    private void writeResultsToScoreDistribution(TreeMap<Integer,AttributeInfo> scroeDistData, int expID, int expIteration,
                                                 int innerIteration, Properties properties, Dataset dataset) throws Exception {

        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Score_Distribution_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        int att_id = 0;
        for(Map.Entry<Integer,AttributeInfo> entry : scroeDistData.entrySet()){
            String metaFeatureName = entry.getValue().getAttributeName();
            Object metaFeatureValueRaw = entry.getValue().getValue();

            //cast results to double
            Double metaFeatureValue = null;
            if (metaFeatureValueRaw instanceof Double) {
                metaFeatureValue = (Double) metaFeatureValueRaw;
            }
            else if(metaFeatureValueRaw instanceof Double){
                metaFeatureValue = ((Double) metaFeatureValueRaw).doubleValue();
            }
            if (Double.isNaN(metaFeatureValue)){
                metaFeatureValue = -1.0;
            }

            //insert to table
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, att_id);
            preparedStmt.setInt (2, expID);
            preparedStmt.setInt (3, expIteration);
            preparedStmt.setInt(4, innerIteration);
            preparedStmt.setString   (5, metaFeatureName);
            preparedStmt.setDouble   (6, metaFeatureValue);

            preparedStmt.execute();
            preparedStmt.close();

            att_id++;
        }
        conn.close();
    }

    private void writeResultsToInstanceMetaFeaturesGroup(HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeInstanceMetaDataInGroup, Properties properties, Dataset dataset) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeInstanceMetaDataInGroup.entrySet()){
            String sql = "insert into tbl_Instances_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, instance_pos,batch_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?, ?, ?)";
            TreeMap<Integer,AttributeInfo> instanceMetaData= outerEntry.getKey();
            int[] instanceInfo = outerEntry.getValue();

            int att_id = 0;
            for(Map.Entry<Integer,AttributeInfo> entry : instanceMetaData.entrySet()){
                String metaFeatureName = entry.getValue().getAttributeName();
                String metaFeatureValue = entry.getValue().getValue().toString();

                try{
                    //insert to table
                    PreparedStatement preparedStmt = conn.prepareStatement(sql);
                    preparedStmt.setInt (1, att_id);
                    preparedStmt.setInt (2, instanceInfo[0]);
                    preparedStmt.setInt (3, instanceInfo[1]);
                    preparedStmt.setInt(4, instanceInfo[2]);
                    preparedStmt.setInt(5, instanceInfo[3]);
                    preparedStmt.setInt (6, instanceInfo[4]);
                    preparedStmt.setString   (7, metaFeatureName);
                    preparedStmt.setString   (8, metaFeatureValue);

                    preparedStmt.execute();
                    preparedStmt.close();

                    att_id++;
                }catch (Exception e){
                    e.printStackTrace();
                    System.out.println("failed insert instance for: (" + att_id+", "+instanceInfo[0]+", "+instanceInfo[1]+", "+instanceInfo[2]
                    + ", "+ instanceInfo[3]+", "+ instanceInfo[4]+", "+ metaFeatureName + ", "+ metaFeatureValue + ")");
                }
            }
        }
        conn.close();
    }

    private void writeResultsToInstanceMetaFeatures(TreeMap<Integer,AttributeInfo> instanceMetaData, int expID, int expIteration,
                                                    int innerIteration, int instancePos, int batchId, Properties properties, Dataset dataset) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Instances_Meta_Data (att_id, exp_id, exp_iteration, inner_iteration_id, instance_pos,batch_id, meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        int att_id = 0;
        for(Map.Entry<Integer,AttributeInfo> entry : instanceMetaData.entrySet()){
            String metaFeatureName = entry.getValue().getAttributeName();
            String metaFeatureValue = entry.getValue().getValue().toString();

            //insert to table
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, att_id);
            preparedStmt.setInt (2, expID);
            preparedStmt.setInt (3, expIteration);
            preparedStmt.setInt(4, innerIteration);
            preparedStmt.setInt(5, instancePos);
            preparedStmt.setInt (6, batchId);
            preparedStmt.setString   (7, metaFeatureName);
            preparedStmt.setString   (8, metaFeatureValue);

            preparedStmt.execute();
            preparedStmt.close();

            att_id++;
        }
        conn.close();
    }

    private void writeResultsToBatchMetaFeaturesGroup(HashMap<TreeMap<Integer, AttributeInfo>, int[]> writeBatchMetaDataInGroup, Properties properties, Dataset dataset) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        for (Map.Entry<TreeMap<Integer, AttributeInfo>, int[]> outerEntry : writeBatchMetaDataInGroup.entrySet()){
            String sql = "insert into tbl_Batches_Meta_Data (att_id, exp_id, exp_iteration, batch_id,meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";
            TreeMap<Integer,AttributeInfo> batchMetaData= outerEntry.getKey();
            int[] batchInfo = outerEntry.getValue();
            int att_id = 0;
            for(Map.Entry<Integer,AttributeInfo> entry : batchMetaData.entrySet()){
                String metaFeatureName = entry.getValue().getAttributeName();
                String metaFeatureValue = entry.getValue().getValue().toString();
                try{
                    //insert to table
                    PreparedStatement preparedStmt = conn.prepareStatement(sql);
                    preparedStmt.setInt(1, att_id);
                    preparedStmt.setInt(2, batchInfo[0]);
                    preparedStmt.setInt(3, batchInfo[1]);
                    preparedStmt.setInt(4, batchInfo[2]);
                    preparedStmt.setString(5, metaFeatureName);
                    preparedStmt.setString(6, metaFeatureValue);

                    preparedStmt.execute();
                    preparedStmt.close();

                    att_id++;
                }
                catch (Exception e){
                    e.printStackTrace();
                    System.out.println("failed insert batch for: (" + att_id+", "+batchInfo[0]+", "+batchInfo[1]+", "+batchInfo[2]
                            + ", "+ metaFeatureName + ", "+ metaFeatureValue + ")");
                }

            }
        }
        conn.close();
    }

    private void writeResultsToBatchesMetaFeatures(TreeMap<Integer,AttributeInfo> batchMetaData, int expID, int innerIteration,
                                                   int batchID, Properties properties, Dataset dataset) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Batches_Meta_Data (att_id, exp_id, exp_iteration, batch_id,meta_feature_name, meta_feature_value) values (?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        int att_id = 0;
        for(Map.Entry<Integer,AttributeInfo> entry : batchMetaData.entrySet()){
            String metaFeatureName = entry.getValue().getAttributeName();
            String metaFeatureValue = entry.getValue().getValue().toString();

            //insert to table
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt(1, att_id);
            preparedStmt.setInt(2, expID);
            preparedStmt.setInt(3, innerIteration);
            preparedStmt.setInt(4, batchID);
            preparedStmt.setString(5, metaFeatureName);
            preparedStmt.setString(6, metaFeatureValue);

            preparedStmt.execute();
            preparedStmt.close();

            att_id++;
        }
        conn.close();
    }

    private void writeToInsertInstancesToBatchGroup(HashMap<ArrayList<Integer>, int[]> writeInsertBatchInGroup, Properties properties) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        for (Map.Entry<ArrayList<Integer>, int[]> outerEntry : writeInsertBatchInGroup.entrySet()){
            String sql = "insert into tbl_Instance_In_Batch(exp_id, exp_iteration, inner_iteration_id,batch_id, instance_pos) values (?, ?, ?, ?, ?)";
            ArrayList<Integer> instancesBatchPos = outerEntry.getKey();
            int[] instanceToBatch = outerEntry.getValue();
            for(Integer instancePosInBatch : instancesBatchPos){
                //insert to table
                int batch_id = instanceToBatch[2]*(-1) - 1;
                PreparedStatement preparedStmt = conn.prepareStatement(sql);
                preparedStmt.setInt (1, instanceToBatch[0]);
                preparedStmt.setInt (2, instanceToBatch[1]);
                preparedStmt.setInt (3, instanceToBatch[2]);
                preparedStmt.setInt (4, batch_id);
                preparedStmt.setInt (5, instancePosInBatch);

                preparedStmt.execute();
                preparedStmt.close();
            }
        }
        conn.close();
    }

    private void writeToInstancesInBatchTbl(int batch_id, int exp_id, int exp_iteration,
                                            ArrayList<Integer> instancesBatchPos, Properties properties)throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Instance_In_Batch(batch_id, exp_id, exp_iteration, instance_id, instance_pos) values (?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));

        for(Integer instancePosInBatch : instancesBatchPos){
            //insert to table
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, batch_id);
            preparedStmt.setInt (2, exp_id);
            preparedStmt.setInt (3, exp_iteration);
            preparedStmt.setInt (4, instancePosInBatch);
            preparedStmt.setInt (5, instancePosInBatch);

            preparedStmt.execute();
            preparedStmt.close();
        }
        conn.close();
    }

    private HashMap<int[], Double> runClassifierOnSampledBatch(int expID, int expIteration, int innerIteration, int batch_id
            , Dataset dataset, Fold testFold, Fold trainFold, HashMap<Integer,Dataset> datasetPartitions,
                                             HashMap<Integer, Integer> batchInstancesToAdd, List<Integer> labeledTrainingSetIndices, Properties properties) throws Exception {

        HashMap<int[], Double> result = new HashMap<>();
        //clone the original dataset
        Dataset clonedDataset = dataset.replicateDataset();
        List<Integer> clonedlabeLedTrainingSetIndices = new ArrayList<>(labeledTrainingSetIndices);
        //run classifier before adding - only for the first batch in the iteration
        if (batch_id == 0){
            AUC aucBeforeAddBatch = new AUC();
            int[] testFoldLabelsBeforeAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
            //Test the entire newly-labeled training set on the test set
            EvaluationInfo evaluationResultsBeforeAdding = runClassifier(properties.getProperty("classifier"),
                    clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                    clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), properties);
            double measureAucBeforeAddBatch = aucBeforeAddBatch.measure
                    (testFoldLabelsBeforeAdding, getSingleClassValueConfidenceScore(evaluationResultsBeforeAdding.getScoreDistributions(),0));
            int[] infoListBefore = new int[5];
            infoListBefore[0] = batch_id;
            infoListBefore[1] = expID;
            infoListBefore[2] = innerIteration;
            infoListBefore[3] = testFoldLabelsBeforeAdding.length;
            infoListBefore[4] = -1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
            result.put(infoListBefore, measureAucBeforeAddBatch);
//            writeToBatchScoreTbl(batch_id,expID, innerIteration, "auc_before_add_batch", measureAucBeforeAddBatch, testFoldLabelsBeforeAdding.length, properties);
        }

        //add batch instances to the cloned dataset
        ArrayList<Integer> instancesClass0 = new ArrayList<>();
        ArrayList<Integer> instancesClass1 = new ArrayList<>();
        for (Map.Entry<Integer,Integer> entry : batchInstancesToAdd.entrySet()){
            int instancePos = entry.getKey();
            int classIndex = entry.getValue();
            if (classIndex == 0){
                instancesClass0.add(instancePos);
            }
            else{
                instancesClass1.add(instancePos);
            }
            clonedlabeLedTrainingSetIndices.add(instancePos);
        }
        clonedDataset.updateInstanceTargetClassValue(instancesClass0, 0);
        clonedDataset.updateInstanceTargetClassValue(instancesClass1, 1);

        //run classifier after adding
        AUC aucAfterAddBatch = new AUC();
        int[] testFoldLabelsAfterAdding = clonedDataset.getTargetClassLabelsByIndex(testFold.getIndices());
        //Test the entire newly-labeled training set on the test set
        EvaluationInfo evaluationResultsAfterAdding = runClassifier(properties.getProperty("classifier"),
                clonedDataset.generateSet(FoldsInfo.foldType.Train,clonedlabeLedTrainingSetIndices),
                clonedDataset.generateSet(FoldsInfo.foldType.Test,testFold.getIndices()), properties);
        double measureAucAfterAddBatch = aucAfterAddBatch.measure
                (testFoldLabelsAfterAdding, getSingleClassValueConfidenceScore(evaluationResultsAfterAdding.getScoreDistributions(),0));
        int[] infoListAfter = new int[5];
        infoListAfter[0] = batch_id;
        infoListAfter[1] = expID;
        infoListAfter[2] = innerIteration;
        infoListAfter[3] = testFoldLabelsAfterAdding.length;
        infoListAfter[4] = 1; //-1="auc_before_add_batch", +1="auc_after_add_batch"
        result.put(infoListAfter, measureAucAfterAddBatch);
//        writeToBatchScoreTbl(batch_id, expID, innerIteration, "auc_after_add_batch", measureAucAfterAddBatch, testFoldLabelsAfterAdding.length, properties);
        return result;
    }

    private void writeToBatchScoreTblGroup(HashMap<int[], Double> writeSampleBatchScoreInGroup, Properties properties) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        for (Map.Entry<int[], Double> outerEntry : writeSampleBatchScoreInGroup.entrySet()){
            String sql = "insert into tbl_Batchs_Score(att_id, batch_id, exp_id, exp_iteration, score_type, score_value, test_set_size) values (?, ?, ?, ?, ?, ?, ?)";
            Double auc = outerEntry.getValue();
            int[] info = outerEntry.getKey();
            int att_id=0;
            //insert to table
            String score_type;
            if (info[4] < 0){
                score_type = "auc_before_add_batch";
            }
            else{
                score_type = "auc_after_add_batch";
            }
            PreparedStatement preparedStmt = conn.prepareStatement(sql);
            preparedStmt.setInt (1, att_id);
            preparedStmt.setInt (2, info[0]);
            preparedStmt.setInt (3, info[1]);
            preparedStmt.setInt (4, info[2]);
            preparedStmt.setString (5, score_type);
            preparedStmt.setDouble (6, auc);
            preparedStmt.setDouble (7, info[3]);

            preparedStmt.execute();
            preparedStmt.close();

            att_id++;

        }
        conn.close();
    }

    private void writeToBatchScoreTbl(int batch_id, int exp_id, int exp_iteration,
                                            String score_type, double score, int test_set_size, Properties properties)throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        String sql = "insert into tbl_Batchs_Score(att_id, batch_id, exp_id, exp_iteration, score_type, score_value, test_set_size) values (?, ?, ?, ?, ?, ?, ?)";

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        int att_id=0;
        //insert to table
        PreparedStatement preparedStmt = conn.prepareStatement(sql);
        preparedStmt.setInt (1, att_id);
        preparedStmt.setInt (2, batch_id);
        preparedStmt.setInt (3, exp_id);
        preparedStmt.setInt (4, exp_iteration);
        preparedStmt.setString (5, score_type);
        preparedStmt.setDouble (6, score);
        preparedStmt.setDouble (7, test_set_size);

        preparedStmt.execute();
        preparedStmt.close();

        att_id++;
        conn.close();
    }
}

