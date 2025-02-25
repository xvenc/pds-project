# PDS - Log file analysis and anomaly detection

**Author:** Václav Korvas (xkorva03)

**Date:** 22-04-2024


The goal of this project is to analyze log files and detect anomalies in these log files. 
This project is based on the Hadoop Distributed File System (HDFS) log file, which is a real-world log file. The log file is provided by the [Loghub](https://github.com/logpai/loghub/tree/master?tab=readme-ov-file). The dataset used in this work is the HDFS version 1 log file. The log file is provided in the `logs/` directory together with the anomaly labels and the template for the log file. The anomaly labels are provided in the `logs/anomaly_label.csv` file and the template for the log file is provided in the `logs/HDFS_template.csv` file. The log file is named `HDFS.log`.

## Instalation

The project was developed using Python 3.11. The project was tested on Ubuntu 20.04.4 LTS and Arch Linux.

First, you need to install the required packages. You can do this by running the following command:

```shell
make install
```

or you can install the packages manually by running the following command:

```shell
pip install -r requirements.txt
```
This will install all the required packages.

The next step is voluntary, because all the necessary log files are present in the zip archive. But if you want to obtain the log file and label file from the original source, you can run the following command:

```shell
wget https://zenodo.org/record/3227177/files/HDFS_1.tar.gz 
```

Then you can run the main script by running the following command:

```shell
python log-monitor.py <arguments> 
```

The next step is to download the necessary datasets. The datasets are available on this google drive [link](https://drive.google.com/drive/folders/1Q6QF2). There are two datasets availabe. One is unbalanced and one is balanced.

## Usage

The main script `log-monitor.py` is used for log file analysis and anomaly detection. The script has the following usage:

```shell

usage: log-monitor.py [-h] [-training TRAINING_FILE] [-testing TESTING_FILE] 
```

If no arguments are provided, the script will use the default values for the arguments. The default values are as follows:

* `-training` - `logs/balanced_train.log`
* `-testing` - `logs/balanced_test.log`

Example of the script usage:

```shell
python log-monitor.py -training logs/balanced_train.log -testing logs/balanced_test.log
```

## Archive structure 

The archive contains 2 main direcotries. The `src/` directory contains the source codes for the project. The `logs/` directory contains the log file, the anomaly labels and the template for the log file. And in the root directory, there is the main script for log file analysis and anomaly detection `log-monitor.py` and the project documentation `xkorva03.pdf`.

## Archive content

* `README.md` - this file
* `log-monitor.py` - main script for log file analysis and anomaly detection
* `src/` - directory with source codes
* `src/dataload.py` - script for log file loading
* `src/model.py` - script for anomaly detection model creation
* `src/preprocess.py` - script for log file preprocessing
* `logs/` - directory with log files
* `logs/HDFS_template.csv` - template for HDFS.log
* `logs/anomaly_label.csv` - anomaly labels for HDFS.log
* `xkorva03.pdf` - project documentation
* `requirements.txt` - file with required packages
* `Makefile` - file with make commands to install the required packages