# loading library
library(car)
library(MASS)
library(corrplot)
library(DescTools)

# loading file

rawData <- read.csv("C:/Users/tk/Desktop/DataScience/Analytics Vidhya/DataSet/Loan Amount/train.csv",na.strings="",stringsAsFactors = TRUE)
predictionData <- read.csv("C:/Users/tk/Desktop/DataScience/Analytics Vidhya/DataSet/Loan Amount/test.csv",na.strings="",stringsAsFactors=TRUE)

colSums(is.na(rawData))
dim(rawData)
dim(predictionData)
View(rawData)
colnames(rawData)
colnames(predictionData)

# creating Loan_Status column in prediction

predictionData$Loan_Status <- NA

#sampling rawData into train & test Data frame

trainRows <- sample(x=1:nrow(rawData),size=0.8*nrow(rawData))

trainData <- rawData[trainRows,]
testData <- rawData[-trainRows,]

dim(trainData)
dim(testData)

# create source column in train,test & prediction data sets

trainData$Source <-"Train"
testData$Source <- "Test"
predictionData$Source <- "Prediction"

dim(trainData)
dim(testData)

dim(predictionData)

# combining all three datasets

fullRaw <- rbind(trainData,testData,predictionData)
dim(fullRaw)

# checking identifier columns
colnames(fullRaw)
# remocing identifier columns

fullRaw$Loan_ID <- NULL
dim(fullRaw)

# checking Null values
summary(fullRaw)

colSums(is.na(fullRaw))





# Missing value imputation

for (i in colnames(fullRaw)){
  if((i!="Loan_Status")|(i!="Source")){
    if((class(fullRaw[,i])=="integer")|(class(fullRaw[,i])=="numeric")){
      tempMedian=median(fullRaw[fullRaw$Source=="Train",i],na.rm=TRUE)
      missingValueRows=is.na(fullRaw[,i])
      fullRaw[missingValueRows,i]=tempMedian
      
    }else{
      tempMode=Mode(fullRaw[fullRaw$Source=="Train",i],na.rm=TRUE)[1]
      missingRows=is.na(fullRaw[,i])
      fullRaw[missingRows,i]=tempMode
    
    }
  }
}

colSums(is.na(fullRaw))

# Levels change fro some columns as per datasets



levels(fullRaw$Loan_Status) <- c(0,1)
fullRaw$Loan_Status=as.integer(fullRaw$Loan_Status)
fullRaw$Loan_Status=ifelse(fullRaw$Loan_Status==1,0,1)
table(fullRaw$Loan_Status)

# lebvels change for dependents variables


levels(fullRaw$Dependents) <-c("No_one","Single","Double","More_3")

# outlier correction


VarForOutliers <-c("ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term")
for (colName in VarForOutliers){
  
  Q1=quantile(fullRaw[fullRaw["Source"]=="Train",colName],probs=0.25)
  Q3=quantile(fullRaw[fullRaw["Source"]=="Train",colName],probs=0.75)
  IQR=Q3-Q1
  LB=Q1-1.5*IQR
  UB=Q3+1.5*IQR
  fullRaw[,colName]=ifelse(fullRaw[,colName]<LB,LB,fullRaw[,colName])
  fullRaw[,colName]=ifelse(fullRaw[,colName]>UB,UB,fullRaw[,colName])
}

# dummy variable creation
factorVars <- sapply(fullRaw,is.factor)
factorVars

dummyVar <- model.matrix(~.,data=fullRaw[,factorVars])
dim(dummyVar)

fullRaw2 <- cbind(fullRaw[,!factorVars],dummyVar[,-1])
dim(fullRaw2)

# dividing data into train,test & prediction & drop Source column
trainDf=subset(fullRaw2,subset=fullRaw2$Source=="Train",select=-Source)
testDf=subset(fullRaw2,subset=fullRaw2$Source=="Test",select=-Source)
predictionDf=subset(fullRaw2,subset=fullRaw2$Source=="Prediction",select=-Source)

dim(trainDf)
dim(testDf)
dim(predictionDf)

############################
# Multicollinearity check
############################



# linear model building for VIF check

M1=lm(Loan_Status~.-Loan_Amount_Term,data=trainDf)
summary(M1)

# Remove variables with VIF > 5

sort(vif(M1), decreasing = TRUE)[1:3]

# we do not have any variable which VIF is greater than 5.So there is not multicollinearity


# Model Building
glm_full <- glm(Loan_Status~.,data=trainDf,family="binomial")
summary(glm_full)

############################
# Model optimization (by selecting ONLY significant variables through step() function)
############################

# Use step() function to remove insignificant variables from the model iteratively

M2=step(glm_full)
summary(M2)

# let us finalize M2 model
pred_test <- predict(M2,testDf,type="response")

head(pred_test)
# change pred_test probability values into 0 & 1
pred_class <- ifelse(pred_test>=0.5,1,0)
head(pred_class)

# confusion matrsix

table(pred_class,testDf$Loan_Status) # prediction & Actual

# Accuracy of the model
sum(diag(table(pred_class,testDf$Loan_Status)))/nrow(testDf)# 83.73%

# TPR = [TP/ Total Actual Positives (1s)]
84/(20+84) # 80.7%

# FPR = [FP/ Total Actual Negatives (0s)]
19/(19+0) # 1%

### Train Prediction & Validation (with 0.5 Cutoff)

# Predict on Train using final model

Train_Pred <- predict(M2,trainDf,type="response")
Train_Class <- ifelse(Train_Pred>=0.5,1,0)
head(Train_Class)

# confusion matrix
table(Train_Class,trainDf$Loan_Status)

# Accuracy
sum(diag(table(Train_Class,trainDf$Loan_Status)))/nrow(trainDf)# 80.2

# Since train & test data has closer accuracy.So our data is not overfitting

# TPR = [TP/ Total Actual Positives (1s)]
331/(90+331) # 78.62%

# FPR = [FP/ Total Actual Negatives (0s)]
63/(63+7) # 9%

# ROC curve
library(ROCR)

ROC_Pred <- prediction(Train_Pred,trainDf$Loan_Status) 

ROC_Curve <- performance(ROC_Pred,"tpr","fpr")

# ROC curve

plot(ROC_Curve)

# selection new cut_off points

Cut_Off_Table <- cbind.data.frame(Cutoff=ROC_Curve@alpha.values[[1]],
                                  FPR=ROC_Curve@x.values[[1]],
                                  TPR=ROC_Curve@y.values[[1]])
View(Cut_Off_Table)

Cut_Off_Table$Difference <- Cut_Off_Table$TPR-Cut_Off_Table$FPR
max(Cut_Off_Table$Difference)

# AUC representation
ROC_AUC <- performance(ROC_Pred,"auc")
ROC_AUC@y.values # 77.6%

# new cutoff point=0.6
Test_Class <- ifelse(pred_test>=0.6,1,0)

# Confusion Matrix
table(Test_Class,testDf$Loan_Status)

# Accuracy
sum(diag(table(Test_Class,testDf$Loan_Status)))/nrow(testDf) # 82.11

# TPR
82/(20+82) # 80.39%

# FPR
19/(19+2) # 9.04%

# prediction on prediction data sets

prediction_predict <- predict(M2,predictionDf,type="response")
head(prediction_predict)

prediction_class <- ifelse(prediction_predict>=0.6,1,0)
table(prediction_class)
prediction<- data.frame(predictionData$Loan_ID)

prediction$Loan_Status <- ifelse(prediction_class==1,"Y","N")

View(prediction)
table(prediction$Loan_Status)
write.csv(prediction,"C:\\Users\\tk\\Desktop\\DataScience\\Analytics Vidhya\\DataSet\\Loan Amount\\prediction.csv")
