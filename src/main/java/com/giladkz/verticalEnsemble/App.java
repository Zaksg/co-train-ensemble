package com.giladkz.verticalEnsemble;

import com.giladkz.verticalEnsemble.CoTrainers.CoTrainerAbstract;
import com.giladkz.verticalEnsemble.CoTrainers.CoTrainerOriginal;
import com.giladkz.verticalEnsemble.CoTrainers.CoTrainingMetaLearning;
import com.giladkz.verticalEnsemble.Data.Dataset;
import com.giladkz.verticalEnsemble.Data.FoldsInfo;
import com.giladkz.verticalEnsemble.Data.Loader;
import com.giladkz.verticalEnsemble.Discretizers.DiscretizerAbstract;
import com.giladkz.verticalEnsemble.Discretizers.EqualRangeBinsDiscretizer;
import com.giladkz.verticalEnsemble.FeatureSelectors.FeatureSelectorInterface;
import com.giladkz.verticalEnsemble.FeatureSelectors.RandomParitionFeatureSelector;
import com.giladkz.verticalEnsemble.ValueFunctions.RandomValues;
import com.giladkz.verticalEnsemble.ValueFunctions.ValueFunctionInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStream;
import java.sql.*;
import java.util.*;
import java.util.Date;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
        Properties properties = new Properties();
        InputStream input = App.class.getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
        Loader loader = new Loader();

        //region Initialization
        DiscretizerAbstract discretizer = new EqualRangeBinsDiscretizer(Integer.parseInt(properties.getProperty("numOfDiscretizationBins")));
        FeatureSelectorInterface featuresSelector = new RandomParitionFeatureSelector();
        ValueFunctionInterface valueFunction = new RandomValues();
        //CoTrainerAbstract coTrainer = new CoTrainingMetaLearning();
        CoTrainerAbstract coTrainer = new CoTrainerOriginal();
        List<Integer> sizeOfLabeledTrainingSet= Arrays.asList( 100);


        //endregion

        File folder = new File(properties.getProperty("inputFilesDirectory"));
        FoldsInfo foldsInfo = InitializeFoldsInfo();

        File[] listOfFiles = folder.listFiles();

        for (File file : listOfFiles) {
            if (file.isFile() && file.getName().endsWith(".arff")) {
                for (int numOfLabeledInstances : sizeOfLabeledTrainingSet) {
                    for (int i=0; i<Integer.parseInt(properties.getProperty("numOfrandomSeeds")); i++) {

                        Date experimentStartDate = new Date();
                        System.out.println(experimentStartDate.toString() + " Beginning analysis of file: " + file.getName() + " with random seed " + i);

                        int expID = getNewExperimentID(file.getName(),coTrainer.toString(),featuresSelector.toString(),valueFunction.toString(), discretizer.toString(), Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")),numOfLabeledInstances, properties);

                        BufferedReader reader = new BufferedReader(new FileReader(file.getAbsolutePath()));
                        Dataset dataset = loader.readArff(reader, i, null, -1, 0.7, foldsInfo);

                        //let the co-training begin
                        //a) generate the feature sets
                        HashMap<Integer, List<Integer>> featureSets = featuresSelector.Get_Feature_Sets(dataset,discretizer,valueFunction,1,2,1000,1,0,dataset.getName(),false, i);

                        Dataset finalDataset = coTrainer.Train_Classifiers(featureSets,dataset,numOfLabeledInstances,Integer.parseInt(properties.getProperty("numOfCoTrainingIterations")),
                                getNumberOfNewInstancesPerClassPerTrainingIteration(dataset.getNumOfClasses(), properties),file.getAbsolutePath(),
                                30000, 1, discretizer, expID,"test",0,0,false, i);


                        Date experimentEndDate = new Date();
                        System.out.println(experimentEndDate.toString() + " Experiment ended");
                    }
                }
            }
        }

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
