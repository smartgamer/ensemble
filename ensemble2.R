####################################
#https://machinelearningmastery.com/machine-learning-ensembles-with-r/

# Combine Model Predictions Into Ensemble Predictions
# 
# The three most popular methods for combining the predictions from different models are:
#   
# 1.Bagging. Building multiple models (typically of the same type) from different subsamples of the training dataset.
# 2.Boosting. Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the chain.
# 3.Stacking. Building multiple models (typically of differing types) and supervisor model that learns how to best combine the predictions of the primary models.

#Before we start building ensembles, let’s define our test set-up.
# Test Dataset
# 
# All of the examples of ensemble predictions in this case study will use the ionosphere dataset.
# 
# This is a dataset available from the UCI Machine Learning Repository. This dataset describes high-frequency antenna returns from high energy particles in the atmosphere and whether the return shows structure or not. The problem is a binary classification that contains 351 instances and 35 numerical attributes.

# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)

# Load the dataset
data(Ionosphere)
dataset <- Ionosphere
dataset <- dataset[,-2]
dataset$V1 <- as.numeric(as.character(dataset$V1))
#Note that the first attribute was a factor (0,1) and has been transformed to be numeric for consistency with all of the other numeric attributes. Also note that the second attribute is a constant and has been removed.
head(dataset)

###
#1. Boosting Algorithms #

# We can look at two of the most popular boosting machine learning algorithms:
#   
#   C5.0
# Stochastic Gradient Boosting
# 
# Below is an example of the C5.0 and Stochastic Gradient Boosting (using the Gradient Boosting Modeling implementation) algorithms in R. Both algorithms include parameters that are not tuned in this example.
# Example of Boosting Algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
# C5.0
set.seed(seed)
fit.c50 <- train(Class~., data=dataset, method="C5.0", metric=metric, trControl=control)
# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Class~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)
# summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)

#Learn more about caret boosting models tree: Boosting Models. http://topepo.github.io/caret/train-models-by-tag.html#Boosting

#2. Bagging Algorithms #

# Let’s look at two of the most popular bagging machine learning algorithms:
#   
# Bagged CART
# Random Forest
# 
# Below is an example of the Bagged CART and Random Forest algorithms in R. Both algorithms include parameters that are not tuned in this example.
# Example of Bagging algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
# Bagged CART
set.seed(seed)
fit.treebag <- train(Class~., data=dataset, method="treebag", metric=metric, trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(Class~., data=dataset, method="rf", metric=metric, trControl=control)
# summarize results
bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)
# We can see that random forest produces a more accurate model with an accuracy of 93.25%
# Learn more about caret bagging model here: Bagging Models. http://topepo.github.io/caret/train-models-by-tag.html#Bagging

# 3. Stacking Algorithms #

# You can combine the predictions of multiple caret models using the caretEnsemble package.
# 
# Given a list of caret models, the caretStack() function can be used to specify a higher-order model to learn how to best combine the predictions of sub-models together.
# 
# Let’s first look at creating 5 sub-models for the ionosphere dataset, specifically:
#   
# Linear Discriminate Analysis (LDA)
# Classification and Regression Trees (CART)
# Logistic Regression (via Generalized Linear Model or GLM)
# k-Nearest Neighbors (kNN)
# Support Vector Machine with a Radial Basis Kernel Function (SVM)
# 
# Below is an example that creates these 5 sub-models. Note the new helpful caretList() function provided by the caretEnsemble package for creating a list of standard caret models.
# Example of Stacking algorithms
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretList(Class~., data=dataset, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)
# We can see that the SVM creates the most accurate model with an accuracy of 94.66%.

# When we combine the predictions of different models using stacking, it is desirable that the predictions made by the sub-models have low correlation. This would suggest that the models are skillful but in different ways, allowing a new classifier to figure out how to get the best from each model for an improved score.
# If the predictions for the sub-models were highly corrected (>0.75) then they would be making the same or very similar predictions most of the time reducing the benefit of combining the predictions.
# correlation between results
modelCor(results)
splom(results)
# We can see that all pairs of predictions have generally low correlation. The two methods with the highest correlation between their predictions are Logistic Regression (GLM) and kNN at 0.517 correlation which is not considered high (>0.75).

# Let’s combine the predictions of the classifiers using a simple linear model.
# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)
# We can see that we have lifted the accuracy to 94.99% which is a small improvement over using SVM alone. This is also an improvement over using random forest alone on the dataset, as observed above.
# We can also use more sophisticated algorithms to combine predictions in an effort to tease out when best to use the different methods. In this case, we can use the random forest algorithm to combine the predictions.

# stack using random forest
set.seed(seed)
stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)
# We can see that this has lifted the accuracy to 96.26% an impressive improvement on SVM alone.



####################################