## This is the dataset of Death rate due to heart failure.There aremany reasons 
## of heart failure which results into death.
## We are require tomake a model which can predict the death rate due to specific reasons

####__________________________________________________________________________
####-------------------------------------------------------------------------

#Load the data

heart_fail<-read.csv("C://Users//tk//Desktop//DataScience//DataSets//Heartfailure//heart_failure_clinical_records_dataset.csv")
heart_fail

head(heart_fail)

#---------------------------------------------------------------------------
### Data Pre processing-------------------------------------------------------
nrow(heart_fail)
ncol(heart_fail)
names(heart_fail)
str(heart_fail)
summary(heart_fail)
table(heart_fail$DEATH_EVENT)

#event rate
96/(96+203)

#checking Null values

sum(is.na(heart_fail))



#---As we see that we do not have missing value----------------------------

colnames(heart_fail)

#We check class of each variable.
#If variable is categorical we require to convert it into factor 

class(heart_fail$age)

class(heart_fail$anaemia)
table(heart_fail$anaemia)
#anaemia is categorical variable.So we require to convert in into factor

heart_fail$anaemia <- as.factor(heart_fail$anaemia)

class(heart_fail$creatinine_phosphokinase)
table(heart_fail$creatinine_phosphokinase)

class(heart_fail$diabetes)
table(heart_fail$diabetes)

heart_fail$diabetes <- as.factor(heart_fail$diabetes)

#changing levels of diabetes

levels(heart_fail$diabetes)[levels(heart_fail$diabetes)=="1"] <-"Yes"
levels(heart_fail$diabetes)[levels(heart_fail$diabetes)=="0"] <- "No"


class(heart_fail$ejection_fraction)
table(heart_fail$ejection_fraction)

#converting high_blood_pressure varaible into facor

class(heart_fail$high_blood_pressure)
table(heart_fail$high_blood_pressure)

heart_fail$high_blood_pressure <- as.factor(heart_fail$high_blood_pressure)

# changing the levels of high_blood_pressure

levels(heart_fail$high_blood_pressure)[levels(heart_fail$high_blood_pressure)=="1"] <- "Yes"
levels(heart_fail$high_blood_pressure)[levels(heart_fail$high_blood_pressure)=="0"] <- "No"

#convertin sex variable into factors

class(heart_fail$sex)
table(heart_fail$sex)
heart_fail$sex <- as.factor(heart_fail$sex)

#changing levels of sex

levels(heart_fail$sex)[levels(heart_fail$sex)=="1"] <- "Male"
levels(heart_fail$sex)[levels(heart_fail$sex)=="0"] <- "Feamle"

#converting smoking variable into factors

class(heart_fail$smoking)
table(heart_fail$smoking)
heart_fail$smoking <- as.factor(heart_fail$smoking)

#changing levels of smoking

levels(heart_fail$smoking)[levels(heart_fail$smoking)=="1"] <- "Yes"
levels(heart_fail$smoking)[levels(heart_fail$smoking)=="0"] <- "No"

class(heart_fail$DEATH_EVENT)
heart_fail$DEATH_EVENT <- as.factor(heart_fail$DEATH_EVENT)

#creating train & test data

#install.packages("caret")
library(caret)
set.seed(1234)
library(dplyr)

train.rows<-createDataPartition(y=heart_fail$DEATH_EVENT,p=0.7,list=F)
train.data <- heart_fail[train.rows,]
length(train.data)
table(train.data$DEATH_EVENT)
68/(143+68)

test.data <- heart_fail[-train.rows,]
table(test.data$DEATH_EVENT)
28/(28+60)

#no of rows in train & test data
nrow(train.data)
nrow(test.data)

#Model Building
help(glm)
glm_model <- glm(DEATH_EVENT~.,family=binomial(link="logit"),data=train.data)
glm_model
summary(glm_model)

#prediction on test data
predict_full_model <- predict(glm_model,test.data,type="response")
summary(predict_full_model)                    
length(predict_full_model)

prediction_full_model <- ifelse(predict_full_model<=0.5,0,1)
table(prediction_full_model)

#building confusion matrix
library(e1071)

confusionMatrix(test.data$DEATH_EVENT,as.factor(prediction_full_model))
(49+21)/(49+21+11+7)
library(ROCR)

#checking model performance analysing tpr & fpr

pre_full_model <-prediction(predict(glm_model),train.data$DEATH_EVENT)

summary(pre_full_model)

perfom_full_model <- performance(pre_full_model,"tpr","fpr")

library(plotROC)

plot(perfom_full_model)

#Model based on significant variables

glm_sign <- glm(DEATH_EVENT~age+ejection_fraction+serum_creatinine+time,data=train.data,
                family=binomial(link="logit"))
summary(glm_sign)

#prediction based on sign variables

glm_sign_predict <- predict(glm_sign,test.data,type="response")
length(glm_sign_predict)

glm_sign_pre <- ifelse(glm_sign_predict<=0.5,0,1)
table(glm_sign_pre)

#Building confusion matrix for significant model variables

confusionMatrix(test.data$DEATH_EVENT,as.factor(glm_sign_pre))

#checking performance anlysing tpr & fpr

pre_sign <- prediction(predict(glm_sign),train.data$DEATH_EVENT)
perfor_pre_sign <- performance(pre_sign,"tpr","fpr")
plot(perfor_pre_sign)
