--dataset meta data
DROP TABLE IF EXISTS `tbl_Dataset_Meta_Data`;
CREATE TABLE `tbl_Dataset_Meta_Data` (
	`dataset_id` int(11) NOT NULL,
	`meta_feature_name` varchar(500) DEFAULT NULL,
	`meta_feature_value` double DEFAULT NULL,
  PRIMARY KEY (`dataset_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--dataset general info
DROP TABLE IF EXISTS `tbl_Dataset`;
CREATE TABLE `tbl_Dataset` (
	`dataset_id` int(11) NOT NULL,
	`dataset_name` varchar(500) DEFAULT NULL,
	`arff_name` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`dataset_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--score distribution meta feature name+value
DROP TABLE IF EXISTS `tbl_Score_Distribution_Meta_Data`;
CREATE TABLE `tbl_Score_Distribution_Meta_Data` (
	`exp_id` int(11) NOT NULL,
	`exp_iteration` int(11) NOT NULL,
	`inner_iteration_id` int(11) NOT NULL,
	`meta_feature_name` varchar(500) DEFAULT NULL,
	`meta_feature_value` double DEFAULT NULL,
  PRIMARY KEY (`exp_id`, `exp_iteration`,`inner_iteration_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--instance meta data
DROP TABLE IF EXISTS `tbl_Instances_Meta_Data`;
CREATE TABLE `tbl_Instances_Meta_Data` (
	`exp_id` int(11) NOT NULL,
	`exp_iteration` int(11) NOT NULL,
	`inner_iteration_id` int(11) NOT NULL,
	`instance_pos` int(11) NOT NULL,
	`meta_feature_name` varchar(500) DEFAULT NULL,
	`meta_feature_value` double DEFAULT NULL,
  PRIMARY KEY (`exp_id`, `exp_iteration`,`inner_iteration_id`, `instance_pos`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--batch meta data
DROP TABLE IF EXISTS `tbl_Batches_Meta_Data`;
CREATE TABLE `tbl_Batches_Meta_Data` (
	`exp_id` int(11) NOT NULL,
	`exp_iteration` int(11) NOT NULL,
	`batch_id` int(11) NOT NULL,
	`meta_feature_name` varchar(500) DEFAULT NULL,
	`meta_feature_value` double DEFAULT NULL,
  PRIMARY KEY (`exp_id`, `exp_iteration`,`batch_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--instances in batch 
DROP TABLE IF EXISTS `tbl_Instance_In_Batch`;
CREATE TABLE `tbl_Instance_In_Batch` (
	`exp_id` int(11) NOT NULL,
	`exp_iteration` int(11) NOT NULL,
	`batch_id` int(11) NOT NULL,
	`instance_pos` int(11) NOT NULL,
	`instance_id` int(11) NOT NULL
  PRIMARY KEY (`exp_id`, `exp_iteration`,`batch_id`,`instance_pos`,`instance_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--batches score 
DROP TABLE IF EXISTS `tbl_Instance_In_Batch`;
CREATE TABLE `tbl_Instance_In_Batch` (
	`exp_id` int(11) NOT NULL,
	`exp_iteration` int(11) NOT NULL,
	`batch_id` int(11) NOT NULL,
	`score_type` varchar(500) DEFAULT NULL,
	`score_value` double DEFAULT NULL,
	`test_set_size` double DEFAULT NULL,
  PRIMARY KEY (`exp_id`, `exp_iteration`,`batch_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;




