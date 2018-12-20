#ensemble methods, R.

#https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/

#Loading the required libraries
library('caret')
#Seeting the random seed
set.seed(1)

#Loading the hackathon dataset
data<-read.csv(url('https://datahack-prod.s3.ap-south-1.amazonaws.com/train_file/train_u6lujuX_CVtuZ9i.csv'))

#Let's see if the structure of dataset data
str(data)


#Does the data contain missing values
sum(is.na(data))


#Imputing missing values using median
preProcValues <- preProcess(data, method = c("medianImpute","center","scale"))
library('RANN')
data_processed <- predict(preProcValues, data)

sum(is.na(data_processed))

#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(data_processed$Loan_Status, p=0.75, list=FALSE)
trainSet <- data_processed[ index,]
testSet <- data_processed[-index,]

#Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)

#Defining the predictors and outcome
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome",
              "CoapplicantIncome")
outcomeName<-'Loan_Status'

#Training the random forest model
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl=fitControl,tuneLength=3)

#Predicting using random forest model
testSet$pred_rf<-predict(object = model_rf,testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,testSet$pred_rf)

#Well, as you can see, we got 0.81 accuracy with the individual random forest model. Let’s see the performance of KNN:

#Training the knn model
model_knn<-train(trainSet[,predictors],trainSet[,outcomeName],method='knn',trControl=fitControl,tuneLength=3)

#Predicting using knn model
testSet$pred_knn<-predict(object = model_knn,testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,testSet$pred_knn)

#It’s great since we are able to get 0.86 accuracy with the individual KNN model. Let’s see the performance of Logistic regression as well before we go on to create ensemble of these three.

#Training the Logistic regression model
model_lr<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm',trControl=fitControl,tuneLength=3)

#Predicting using knn model
testSet$pred_lr<-predict(object = model_lr,testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,testSet$pred_lr)

#And the logistic regression also gives us the accuracy of 0.86.


##
# Now, let’s try out different ways of forming an ensemble with these models as we have discussed:
#   
#   Averaging: In this, we’ll average the predictions from the three models. Since the predictions are either ‘Y’ or ‘N’, averaging doesn’t make much sense for this binary classification. However, we can do averaging on the probabilities of observations to be in wither of these binary classes.

#Predicting the probabilities
testSet$pred_rf_prob<-predict(object = model_rf,testSet[,predictors],type='prob')
testSet$pred_knn_prob<-predict(object = model_knn,testSet[,predictors],type='prob')
testSet$pred_lr_prob<-predict(object = model_lr,testSet[,predictors],type='prob')

#Taking average of predictions
testSet$pred_avg<-(testSet$pred_rf_prob$Y+testSet$pred_knn_prob$Y+testSet$pred_lr_prob$Y)/3

#Splitting into binary classes at 0.5
testSet$pred_avg<-as.factor(ifelse(testSet$pred_avg>0.5,'Y','N'))

# Majority Voting: In majority voting, we’ll assign the prediction for the observation as predicted by the majority of models. Since we have three models for a binary classification task, a tie is not possible.

#The majority vote
testSet$pred_majority<-as.factor(ifelse(testSet$pred_rf=='Y' & testSet$pred_knn=='Y','Y',ifelse(testSet$pred_rf=='Y' & testSet$pred_lr=='Y','Y',ifelse(testSet$pred_knn=='Y' & testSet$pred_lr=='Y','Y','N'))))

# Weighted Average: Instead of taking simple average, we can take weighted average. Generally, the weights of predictions are high for more accurate models. Let’s assign 0.5 to logistic regression and 0.25 to KNN and random forest each.

#Taking weighted average of predictions
testSet$pred_weighted_avg<-(testSet$pred_rf_prob$Y*0.25)+(testSet$pred_knn_prob$Y*0.25)+(testSet$pred_lr_prob$Y*0.5)

#Splitting into binary classes at 0.5
testSet$pred_weighted_avg<-as.factor(ifelse(testSet$pred_weighted_avg>0.5,'Y','N'))

#Before proceeding further, I would like you to recall about two important criteria that we previously discussed on individual model accuracy and inter-model prediction correlation which must be fulfilled. In the above ensembles, I have skipped checking for the correlation between the predictions of the three models. I have randomly chosen these three models for a demonstration of the concepts. If the predictions are highly correlated, then using these three might not give better results than individual models. But you got the point. Right?

# So far, we have used simple formulas at the top layer. Instead, we can use another machine learning model which is essentially what stacking is. We can use linear regression for making a linear formula for making the predictions in regression problem for mapping bottom layer model predictions to the outcome or logistic regression similarly in case of classification problem.
# 
# Moreover, we don’t need to restrict ourselves here, we can also use more complex models like GBM, neural nets to develop a non-linear mapping from the predictions of bottom layer models to the outcome.
# 
# On the same example let’s try applying logistic regression and GBM as top layer models. Remember, the following steps that we’ll take:
#   
#   Train the individual base layer models on training data.
# Predict using each base layer model for training data and test data.
# Now train the top layer model again on the predictions of the bottom layer models that has been made on the training data.
# Finally, predict using the top layer model with the predictions of bottom layer models that has been made for testing data.
# 
# One extremely important thing to note in step 2 is that you should always make out of bag predictions for the training data, otherwise the importance of the base layer models will only be a function of how well a base layer model can recall the training data.
# 
# Even, most of the steps have been already done previously, but I’ll walk you through the steps one by one again.



# Step 1: Train the individual base layer models on training data
#Defining the training control
fitControl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = 'final', # To save out of fold predictions for best parameter combinantions
  classProbs = T # To save the class probabilities of the out of fold predictions
)

#Defining the predictors and outcome
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome",
              "CoapplicantIncome")
outcomeName<-'Loan_Status'

#Training the random forest model
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl=fitControl,tuneLength=3
                
#Training the knn model
model_knn<-train(trainSet[,predictors],trainSet[,outcomeName],method='knn',trControl=fitControl,tuneLength=3)
                
#Training the logistic regression model
model_lr<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm',trControl=fitControl,tuneLength=3)

#Step 2: Predict using each base layer model for training data and test data
#Predicting the out of fold prediction probabilities for training data
trainSet$OOF_pred_rf<-model_rf$pred$Y[order(model_rf$pred$rowIndex)]
trainSet$OOF_pred_knn<-model_knn$pred$Y[order(model_knn$pred$rowIndex)]
trainSet$OOF_pred_lr<-model_lr$pred$Y[order(model_lr$pred$rowIndex)]

#Predicting probabilities for the test data
testSet$OOF_pred_rf<-predict(model_rf,testSet[predictors],type='prob')$Y
testSet$OOF_pred_knn<-predict(model_knn,testSet[predictors],type='prob')$Y
testSet$OOF_pred_lr<-predict(model_lr,testSet[predictors],type='prob')$Y

#Step 3: Now train the top layer model again on the predictions of the bottom layer models that has been made on the training data

#Predictors for top layer models 
predictors_top<-c('OOF_pred_rf','OOF_pred_knn','OOF_pred_lr') 

#GBM as top layer model 
model_gbm<- 
  train(trainSet[,predictors_top],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=3)

#Logistic regression as top layer model
model_glm<-
  train(trainSet[,predictors_top],trainSet[,outcomeName],method='glm',trControl=fitControl,tuneLength=3)

#Step 4: Finally, predict using the top layer model with the predictions of bottom layer models that has been made for testing data

#predict using GBM top layer model
testSet$gbm_stacked<-predict(model_gbm,testSet[,predictors_top])

#predict using logictic regression top layer model
testSet$glm_stacked<-predict(model_glm,testSet[,predictors_top])

#5. Additional resources

# By now, you might have developed an in-depth conceptual as well as practical knowledge of ensembling. I would like to encourage you to practice this on machine learning hackathons on Analytics Vidhya, which you can find here.








