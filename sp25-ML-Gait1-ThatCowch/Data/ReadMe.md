## Data Context

The data capture correspond to individual wearing an IMU on their shin while walking on Centennial campus. Their gait motion was captured while they were standing, walking on solid even terrain and softer uneven terrain (grass in our case), and climbing up and down stairs. This data was captured to train a gait recognition model to aid on the development of robotic prosthesis.

## Dataset Description

Here is a brief description of the data files:

  - ".x.v" files contain the xyz accelerometers and xyz gyroscope measurements from the lower limb.
  - ".x.t" files contain the time stamps for the accelerometer and gyroscope measurements. The units are in seconds and the sampling rate is 40 Hz.
  - ".y.v" files contain the labels. (0) indicates standing or walking in solid ground, (1) indicates going down the stairs, (2) indicates going up the stairs, and (3) indicates walking on grass.
  - ".y.t" files contain the time stamps for the labels. The units are in seconds and the sampling rates is 10 Hz.

The dataset contents multiple sessions some of which are coming from the same subject. We have renamed all these sessions to trials so you cannot identify the origin of the session. The training folder contains the data files for all the trials considered for training. The test set 1 contains trials corresponds to sessions from subjects that have already been observed. The test set 2 contains trails corresponds to sessions from subjects that have not been observed before. Both test folders are missing the 'y.v' label files. Those are the files that you will need to generate. Note that the two datasets aim to evaluation two different scenarios for generalization of your model.

The data set is imbalanced. Here are some suggestions for handling imbalance:

  1. Make sure you create a validation set that is also balanced in order to better represent the type of testing data you will get.
  2. You can modify your loss function to include weights that compensate for the imbalance distributions. A quick search online would give you some hints on how to do this.
  3. When doing data augmentation, you can make sure your training data is balanced by getting more replications (with some deformation / noise) for those classes that have fewer samples.
  4. You can also apply a subsampling approach when creating your batches which includes all the data for the smaller datasets but selects a smaller proportion from the classes with most instances (in order to keep the number per class about the same).