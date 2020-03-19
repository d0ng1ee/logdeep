# logdeep

## Introduction

LogDeep is an open source deeplearning-based log analysis toolkit for automated anomaly detection.

![Framework of Anomaly Detection](data/semantic_vec.pdf)

## Major features

- Modular Design

- Support multi log event features out of box

- State of the art(Including resluts from deeplog,loganomaly,robustlog...)

## Models

| Model | Paper reference |
| :--- | :--- |
|DeepLog| [**CCS'17**] [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf)|
|LogAnomaly| [**IJCAI'19**] [LogAnomaly: UnsupervisedDetectionof SequentialandQuantitativeAnomaliesinUnstructuredLogs](https://www.ijcai.org/Proceedings/2019/658)|
|RobustLog| [**FSE'19**] [RobustLog-BasedAnomalyDetectiononUnstableLogData](https://dl.acm.org/doi/10.1145/3338906.3338931)

## Benchmark results

|       |            | HDFS |     |
| :----:|:----:|:----:|:----:|:----:|
| **Model** | **feature** | **Precision** | **Recall** | **F1** |
| deeplog(unsupervised)| seq |0.953 | 0.948 | 0.95 |
| loganomaly(unsupervised) | seq+quan| | | |
| robustlog(supervised)| semantic | | | |
