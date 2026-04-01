# CS 1 Report
## Discuss the functions and parameters that you will vary for implementing Logistic Regression, SVM, and Decision Trees.
- **Logistic Regression**
    - Here, we implemented a grid search to try a variety of solvers and parameters
    - For parameters, we varied C and the L1 ratio 
        - Tested factors of 10 for C from 0.1 to 100
        - Tested 0 and 1 for the L1 ratio
    - For the solver function, we tried all six different solvers that sklearn supports
- **SVM**
    - We also did a grid search here to find the best combination
    - For parameters we varied C and gamma
        - Varied C by factors of 10 from 0.1 to 100
        - Tried gamma as scale, auto, 0.1, or 0.01
    - For the kernel function we tested rbf, poly, and sigmoid
- **Decision Trees**
    - We also did a grid search for the best parameter here
    - For parameters we varied max depth, minimum samples split, minimum samples leaf, and maximum features.
        - For the max depth we did 3, 5, 10, 20, and None
        - For minimum samples split we did 2, 5, 10, and 20
        - For minimum samples leaf we did 1, 2, 5, 10
        - For maximum features we tried sqrt, log2, and none
    - For functions we varied the criterea and class weight
        - We tried gini and entropy criteria
        - We tried None and balanced class weight
    - Decision Trees ran very fast so we could test many parameters to find the best
## Explain the data preprocessing algorithms you adopted and their impacts.

## Explain your hyperparameter tuning process and why it works.
## The reason about your final choice of classification model and its related data preprocessing and hyperparameter tuning algorithms.
    We did not manually choose a classification model, but instead ran each of the models on the training set individually, then just automatically used the one with the highest accuracy. In order to minimize overfitting, each model was run using grid search cross validation across all available hyperparameters, so it would optimize each model individually and compare only the best specimens. We used a k-fold with k=5 within each cross-validator, in order to at least attempt to get as broad a range of validation data as possible

## Insightfulness and clarity of your observations and discussions. (Please be free to add the approaches you tried but failed before arriving at the best solution.)