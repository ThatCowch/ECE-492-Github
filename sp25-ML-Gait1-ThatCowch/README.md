# Gait Recognition Project 1

In this project, you will be building an MLP using hand-crafted features. Your goal will be recognition of human gait while walking on hard and soft terrains, and climbing up and down stairs. The input to your models will be hand-crafted features from IMU devices. As part of this project, you will become familiar with the data and also perform some hyper-parameter tuning manually or using a grid search.

Here is a brief description of the notebooks:
- `ProjC1.1 - EDA.ipynb` - Performs some exploratory analysis on the data. It is useful to look at the results since this can give your choice on other sections of the project.
- `ProjC1.2 - Baseline RF.ipynb` - It trains a simple Random Forest model for classification using the hand-crafted features. We report the results using the validation set. We will use the test set in the next phase of the project. Useful to have an idea of how good should your models perform.
- `ProjC1.3 - Baseline MLP.ipynb` - It uses an MLP for classification. This is the only portion that will be graded and it requires some hyper-parameter selection.
