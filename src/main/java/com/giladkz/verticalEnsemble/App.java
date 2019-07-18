package com.giladkz.verticalEnsemble;

import com.giladkz.verticalEnsemble.CoTrainers.CoTrainMetaModelLoded;
import com.giladkz.verticalEnsemble.CoTrainers.CoTrainerAbstract;
import com.giladkz.verticalEnsemble.CoTrainers.CoTrainerOriginal;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Data.FoldsInfo;
import com.giladkz.verticalEnsemble.Data.Loader;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.Discretizers.EqualRangeBinsDiscretizer;
import com.giladkz.verticalEnsemble.FeatureSelectors.FeatureSelectorInterface;
import com.giladkz.verticalEnsemble.FeatureSelectors.RandomParitionFeatureSelector;
import com.giladkz.verticalEnsemble.StatisticsCalculations.AUC;
import com.giladkz.verticalEnsemble.ValueFunctions.RandomValues;
import com.giladkz.verticalEnsemble.ValueFunctions.ValueFunctionInterface;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.sql.*;
import java.util.*;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class App 
{
    public static void main( String[] args ) throws Exception
    {
        Properties properties = new Properties();
        InputStream input = App.class.getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
        Loader loader = new Loader();

        buildDBTables(properties);
        buildMetaLearnDBTables(properties);
        //region Initialization
        DiscretizerAbstract discretizer = new EqualRangeBinsDiscretizer(Integer.parseInt(properties.getProperty("numOfDiscretizationBins")));
        FeatureSelectorInterface featuresSelector = new RandomParitionFeatureSelector();
        ValueFunctionInterface valueFunction = new RandomValues();
        //CoTrainerAbstract coTrainer = new CoTrainingMetaLearning();
        CoTrainerAbstract coTrainer_original = new CoTrainerOriginal();
        CoTrainerAbstract coTrainer_meta_model = new CoTrainMetaModelLoded();
        List<Integer> sizeOfLabeledTrainingSet= Arrays.asList(100);
        //endregion

        File folder = new File(properties.getProperty("inputFilesDirectory"));
        FoldsInfo foldsInfo = InitializeFoldsInfo();

        File [] listOfFiles;
        File[] listOfFilesTMP = folder.listFiles();
        List<File> listFilesBeforeShuffle = Arrays.asList(listOfFilesTMP);
        Collections.shuffle(listFilesBeforeShuffle);
        listOfFiles = (File[])listFilesBeforeShuffle.toArray();


        //String[] toDoDatasets = {"cardiography_new.arff"};
        String[] toDoDatasets = {"german_credit.arff", "ailerons.arff", "cardiography_new.arff"
                , "contraceptive.arff", "cpu_act.arff"
                , "delta_elevators.arff", "puma32H.arff", "puma8NH.arff", "seismic-bumps.arff"
                , "space_ga.arff", "wind.arff"}; // "bank-full.arff",
        if(args.length > 0){
            toDoDatasets = args;
        }
        String[] doneDatasets = {"german_credit.arff"};

        //double auc_check = checkAucClac();

        for (File file : listOfFiles) {
            if (file.isFile() && file.getName().endsWith(".arff") /*&& !Arrays.asList(doneDatasets).contains(file.getName())*/
                    && Arrays.asList(toDoDatasets).contains(file.getName())) {

                for (int numOfLabeledInstances : sizeOfLabeledTrainingSet) {
                    int numOfRuns = Integer.parseInt(properties.getProperty("numOfrandomSeeds"));
                    ArrayList<Runnable> tasks = new ArrayList<>();
                    ExecutorService executorService = Executors.newFixedThreadPool(numOfRuns);
                    ArrayList<ArrayList<Integer>> exp_ids = getExpIds(numOfRuns, file.getName(),coTrainer_original.toString(),featuresSelector.toString(),valueFunction.toString(), discretizer.toString(), Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")),numOfLabeledInstances, properties);
                    //runDatasetSeed(0, file, numOfLabeledInstances, coTrainer_original, coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo, properties);

                    for (int task_i = 0; task_i < numOfRuns; task_i++) {
                        final int taks_index = task_i;
                        Runnable task_temp = () -> {
                            try {
                                runDatasetSeed(taks_index, file, numOfLabeledInstances, coTrainer_original
                                        , coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo
                                        , properties, exp_ids.get(taks_index).get(0), exp_ids.get(taks_index).get(1));
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        };
                        tasks.add(task_temp);
                    }
                    for (int task_i = 0; task_i < numOfRuns; task_i++) {
                        executorService.submit(tasks.get(task_i));
                    }
                    executorService.shutdownNow();

/*                    Runnable task1 = () -> {
                        try {
                            runDatasetSeed(0, file, numOfLabeledInstances, coTrainer_original, coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo, properties, exp_ids.get(0).get(0), exp_ids.get(0).get(1));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    };
                    Runnable task2 = () -> {
                        try {
                            //TimeUnit.MILLISECONDS.sleep(1500);
                            runDatasetSeed(1, file, numOfLabeledInstances, coTrainer_original, coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo, properties, exp_ids.get(1).get(0), exp_ids.get(1).get(1));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    };
                    Runnable task3 = () -> {
                        try {
                            //TimeUnit.MILLISECONDS.sleep(3000);
                            runDatasetSeed(2, file, numOfLabeledInstances, coTrainer_original, coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo, properties, exp_ids.get(2).get(0), exp_ids.get(2).get(1));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    };
                    Runnable task4 = () -> {
                        try {
                            //TimeUnit.MILLISECONDS.sleep(4500);
                            runDatasetSeed(3, file, numOfLabeledInstances, coTrainer_original, coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo, properties, exp_ids.get(3).get(0), exp_ids.get(3).get(1));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    };
                    Runnable task5 = () -> {
                        try {
                            //TimeUnit.MILLISECONDS.sleep(6000);
                            runDatasetSeed(4, file, numOfLabeledInstances, coTrainer_original, coTrainer_meta_model, discretizer, featuresSelector, valueFunction, loader, foldsInfo, properties, exp_ids.get(4).get(0), exp_ids.get(4).get(1));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    };
                    ExecutorService executorService = Executors.newFixedThreadPool(5);
                    executorService.submit(task1);
                    executorService.submit(task2);
                    executorService.submit(task3);
                    executorService.submit(task4);
                    executorService.submit(task5);
                    executorService.shutdownNow();
                    */
                }
            }
        }

    }

    private static ArrayList<ArrayList<Integer>> getExpIds(int runsPerDS, String arff_name, String co_trainer, String feature_selector,
                                                           String value_function, String discretizer, int num_of_training_iterations
            , int labeled_training_set_size, Properties properties) throws Exception{
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        for (int i = 0; i < runsPerDS; i++) {
            ArrayList<Integer> temp = new ArrayList<>();
            int expID_original = getNewExperimentID(arff_name,co_trainer,feature_selector,value_function, discretizer, num_of_training_iterations,labeled_training_set_size, properties);
            int expID_meta_model = getNewExperimentID(arff_name,co_trainer,feature_selector,value_function, discretizer, num_of_training_iterations,labeled_training_set_size, properties);
            temp.add(expID_original);
            temp.add(expID_meta_model);
            res.add(temp);
        }
        return res;
    }

    private static void runDatasetSeed(int i, File file, int numOfLabeledInstances, CoTrainerAbstract coTrainer_original, CoTrainerAbstract coTrainer_meta_model, DiscretizerAbstract discretizer, FeatureSelectorInterface featuresSelector, ValueFunctionInterface valueFunction, Loader loader, FoldsInfo foldsInfo, Properties properties, Integer expID_original, Integer expID_meta_model) throws Exception{

        BufferedReader reader = new BufferedReader(new FileReader(file.getAbsolutePath()));
        try{
            Dataset dataset = loader.readArff(reader, i, null, -1, 0.7, foldsInfo);
            Dataset dataset_meta_model = dataset.replicateDataset();
            //let the co-training begin
            //a) generate the feature sets
            HashMap<Integer, List<Integer>> featureSets = featuresSelector.Get_Feature_Sets(dataset,discretizer,valueFunction,1,2,1000,1,0,dataset.getName(),false, i);

            Dataset finalDataset_otiginal = coTrainer_original.Train_Classifiers(featureSets,dataset,numOfLabeledInstances,Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")), getNumberOfNewInstancesPerClassPerTrainingIteration(dataset.getNumOfClasses(), properties),file.getAbsolutePath(),30000, 1, discretizer, expID_original,"test",0,0,false, i);

            System.out.println("Original model done with exp id: "+expID_original+". Start meta model with exp id: "+ expID_meta_model);
            Dataset finalDataset_meta_model = coTrainer_meta_model.Train_Classifiers(featureSets,dataset_meta_model,numOfLabeledInstances,Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")), getNumberOfNewInstancesPerClassPerTrainingIteration(dataset_meta_model.getNumOfClasses(), properties),file.getAbsolutePath(),30000, 1, discretizer, expID_meta_model,"test",0,0,false, i);


            Date experimentEndDate = new Date();
            System.out.println(experimentEndDate.toString() + " Experiment ended");
        }catch (Exception e){
            e.printStackTrace();
            System.out.println("Failed running dataset: " + file.getName());
        }
    }

    private static double checkAucClac() throws FileNotFoundException {
        AUC auc = new AUC();
        ArffLoader loader1 = new ArffLoader();
        ArffLoader loader2 = new ArffLoader();
        try{
            loader1.setFile(new File("/Users/guyz/Documents/CoTrainingVerticalEnsemble/weka testing/german_credit_100.arff"));
            loader2.setFile(new File("/Users/guyz/Documents/CoTrainingVerticalEnsemble/weka testing/german_credit_rest.arff"));
            Instances train = loader1.getDataSet();
            Instances test = loader2.getDataSet();
            train.setClassIndex(train.numAttributes() - 1);
            test.setClassIndex(test.numAttributes() - 1);

            int[] truth = new int[test.size()];
            for (int i = 0; i < test.size(); i++) {
                truth[i] = (int)test.get(i).classValue();
            }
            
            
            RandomForest randomForest = new RandomForest();
            randomForest.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            double[] ab = new double[test.size()];
            for (int i=0; i<test.size(); i++) {
                Instance testInstance = test.get(i);
                double[] score = randomForest.distributionForInstance(testInstance);
                ab[i] = score[1];
            }
            double[] a = eval.evaluateModel(randomForest, test);
            ThresholdCurve tc = new ThresholdCurve();
            int classIndex = 1;
            Instances result = tc.getCurve(eval.predictions(), classIndex);
            double auc_code = auc.measure(truth, ab);
            double auc_weka = tc.getROCArea(result);
            return auc_code - auc_weka;
        }
        catch (Exception e){
            e.printStackTrace();
        }

        return 0.0;
    }

    private static void buildDBTables(Properties properties)  throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        Statement stmt = conn.createStatement();
        String sqlTbl1 = "CREATE TABLE if not exists tbl_Batches_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl1);
        String sqlTbl2 = "CREATE TABLE if not exists tbl_Batchs_Score (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  score_type varchar(500) NOT NULL," +
                "  score_value double DEFAULT NULL," +
                "  test_set_size double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,score_type)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl2);

        String sqlTbl3 = "CREATE TABLE if not exists tbl_Co_Training_Added_Samples (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  weight float NOT NULL," +
                "  inner_iteration int(11) NOT NULL," +
                "  classifier_id int(11) NOT NULL," +
                "  sample_pos int(11) NOT NULL," +
                "  presumed_class int(11) DEFAULT NULL," +
                "  is_correct tinyint(4) DEFAULT NULL," +
                "  certainty float DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,weight,exp_iteration,sample_pos,classifier_id,inner_iteration)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl3);

        String sqlTbl4 = "CREATE TABLE if not exists tbl_Dataset (" +
                "  dataset_id int(11) NOT NULL," +
                "  dataset_name varchar(500) DEFAULT NULL," +
                "  arff_name varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (dataset_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl4);

        String sqlTbl5 = "CREATE TABLE if not exists tbl_Dataset_Meta_Data (" +
                "  dataset_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) DEFAULT NULL," +
                "  meta_feature_value double DEFAULT NULL," +
                "  PRIMARY KEY (dataset_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl5);

        String sqlTbl6 = "CREATE TABLE if not exists tbl_Experiments (" +
                "  exp_id int(11) NOT NULL," +
                "  arff_name varchar(500) DEFAULT NULL," +
                "  start_date datetime DEFAULT NULL," +
                "  co_trainer varchar(500) DEFAULT NULL," +
                "  feature_selector varchar(500) DEFAULT NULL," +
                "  value_function varchar(500) DEFAULT NULL," +
                "  discretizer varchar(500) DEFAULT NULL," +
                "  num_of_training_iterations int(11) DEFAULT NULL," +
                "  classifier varchar(500) DEFAULT NULL," +
                "  labeled_training_set_size int(11) DEFAULT NULL," +
                "  item_insertion_policy varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (exp_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl6);

        String sqlTbl7 = "CREATE TABLE if not exists tbl_Instance_In_Batch (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  PRIMARY KEY (exp_id,exp_iteration,inner_iteration_id,batch_id,instance_pos)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl7);

        String sqlTbl8 = "CREATE TABLE if not exists tbl_Instances_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,instance_pos,meta_feature_name,batch_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl8);

        String sqlTbl9 = "CREATE TABLE if not exists tbl_Meta_Data_Added_Batches (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  total_batch_size int(11) DEFAULT NULL," +
                "  num_of_class_1_assigned_labels int(11) DEFAULT NULL," +
                "  num_of_class_2_assigned_labels int(11) DEFAULT NULL," +
                "  averaging_score_absolue double DEFAULT NULL," +
                "  multiplication_score_absolute double DEFAULT NULL," +
                "  unified_score_absolute double DEFAULT NULL," +
                "  averaging_score_delta double DEFAULT NULL," +
                "  multiplication_score_delta double DEFAULT NULL," +
                "  unified_score_delta double DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,exp_iteration,inner_iteration,batch_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl9);

        String sqlTbl10 = "CREATE TABLE if not exists tbl_Meta_Data_Batch_Info (" +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  assigned_class int(11) DEFAULT NULL," +
                "  true_class int(11) DEFAULT NULL," +
                "  classifier_1_score double DEFAULT NULL," +
                "  classifier_2_score double DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,exp_iteration,inner_iteration,batch_id,instance_pos)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl10);

        String sqlTbl11 = "CREATE TABLE if not exists tbl_Score_Distribution_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl11);

        String sqlTbl12 = "CREATE TABLE if not exists tbl_Test_Set_Evaluation_Results (" +
                "  exp_id int(11) NOT NULL," +
                "  iteration_id int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  classification_calculation_method varchar(500) NOT NULL," +
                "  metric_name varchar(500) NOT NULL," +
                "  ensemble_size int(11) NOT NULL," +
                "  confidence_level double NOT NULL," +
                "  value double DEFAULT NULL," +
                "  PRIMARY KEY (exp_id,iteration_id,inner_iteration_id,classification_calculation_method,metric_name,ensemble_size,confidence_level)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl12);
        conn.close();

        System.out.println("databases created");
    }

    private static void buildMetaLearnDBTables(Properties properties) throws Exception{
        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);
        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        Statement stmt = conn.createStatement();

        String sqlTbl_1 = "CREATE TABLE if not exists tbl_meta_learn_Batches_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_1);

        String sqlTbl_2 = "CREATE TABLE if not exists tbl_meta_learn_Batchs_Score (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  score_type varchar(500) NOT NULL," +
                "  score_value double DEFAULT NULL," +
                "  test_set_size double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,batch_id,score_type)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_2);

        String sqlTbl_3 = "CREATE TABLE if not exists tbl_meta_learn_Instances_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  instance_pos int(11) NOT NULL," +
                "  batch_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value varchar(500) DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,instance_pos,meta_feature_name,batch_id)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_3);

        String sqlTbl_4 = "CREATE TABLE if not exists tbl_meta_learn_Score_Distribution_Meta_Data (" +
                "  att_id int(11) NOT NULL," +
                "  exp_id int(11) NOT NULL," +
                "  exp_iteration int(11) NOT NULL," +
                "  inner_iteration_id int(11) NOT NULL," +
                "  meta_feature_name varchar(500) NOT NULL," +
                "  meta_feature_value double DEFAULT NULL," +
                "  PRIMARY KEY (att_id,exp_id,exp_iteration,inner_iteration_id,meta_feature_name)" +
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8;";
        stmt.executeUpdate(sqlTbl_4);

        conn.close();

        System.out.println("meta learn databases created");
    }

    /**
     * Initializes the object containing the information regarding how the original dataset need to be partitioned into train, validaion
     * and test sets. Also determines the number of folds for each type.
     * @return
     * @throws Exception
     */
    private static FoldsInfo InitializeFoldsInfo() throws Exception {
        FoldsInfo fi = new FoldsInfo(1,0,1,0.7,-1,0,0,0.3,-1,true, FoldsInfo.foldType.Test);
        return fi;
    }

    private static HashMap<Integer,Integer> getNumberOfNewInstancesPerClassPerTrainingIteration(int numOfClasses, Properties properties) {
        HashMap<Integer,Integer> mapToReturn = new HashMap<>();
        for (int i=0; i<numOfClasses; i++) {
            mapToReturn.put(i, Integer.parseInt(properties.getProperty("numOfInstancesToAddPerIterationPerClass")));
        }
        return mapToReturn;
    }

    private static int getNewExperimentID(String arff_name, String co_trainer, String feature_selector,
                                      String value_function, String discretizer, int num_of_training_iterations, int labeled_training_set_size, Properties properties) throws Exception {
        int exp_id;
        //tbl_Clustering_Runs
        String sql = "select IFNULL ( max(exp_id), 0 ) as idx from tbl_Experiments";

        String myDriver = properties.getProperty("JDBC_DRIVER");
        String myUrl = properties.getProperty("DatabaseUrl");
        Class.forName(myDriver);

        Connection conn = DriverManager.getConnection(myUrl, properties.getProperty("DBUser"), properties.getProperty("DBPassword"));
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery(sql);
        if (rs.next()){
            exp_id = rs.getInt("idx");
        }
        else {
            throw new Exception("no run id created");
        }
        rs.close();
        stmt.close();

        Date date = new Date();
        sql = "insert into tbl_Experiments (exp_id, arff_name, start_date, co_trainer, feature_selector, value_function, discretizer, num_of_training_iterations, classifier, labeled_training_set_size,item_insertion_policy) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        PreparedStatement preparedStmt = conn.prepareStatement(sql);
        preparedStmt.setInt (1, exp_id+1);
        preparedStmt.setString (2, arff_name);
        preparedStmt.setTimestamp   (3, new java.sql.Timestamp(new java.util.Date().getTime()));
        preparedStmt.setString(4, co_trainer);
        preparedStmt.setString(5, feature_selector);
        preparedStmt.setString(6, value_function);
        preparedStmt.setString(7, discretizer);
        preparedStmt.setInt(8, num_of_training_iterations);
        preparedStmt.setString(9, properties.getProperty("classifier"));
        preparedStmt.setInt(10, labeled_training_set_size);
        preparedStmt.setString(11, "fixed");
        preparedStmt.execute();

        conn.close();


        return exp_id + 1;
    }




}
