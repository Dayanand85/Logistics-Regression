# Package loading
library(car)
library(ROCR)

# Set Working Directory
setwd("C:/Users/Rahul/Documents/IMR/Data_Logistic_Regression/")

# Read the data files
# FullRaw = read.csv("BankCreditCard.csv", stringsAsFactors = TRUE)
FullRaw = read.csv("C:\\Users\\tk\\Desktop\\DataScience\\R\\R_Material\\R_Material\\3_Logistic_Regression\\BankCreditCard.csv",stringsAsFactors = TRUE)
# View the data
#View(FullRaw)

# Check for NAs
colSums(is.na(FullRaw))

# Check the summary of the data
summary(FullRaw) 


############################
# Sampling: Divide the data into Train and Testset
############################
# We are conducting sampling  right at the beginning to avoid 
# information leak from training data to testing data. So, we
# will identify training and testing rows right at the beginning, and
# mark them using "Source" column.

set.seed(981) # This is used to reproduce the SAME composition of the sample EVERYTIME
RowNumbers = sample(x = 1:nrow(FullRaw), size = 0.70*nrow(FullRaw))
# head(RowNumbers)
TrainRaw = FullRaw[RowNumbers, ] # Train
TestRaw = FullRaw[-RowNumbers, ] # Testset

# Mark train rows and test rows
TrainRaw$Source = "Train"
TestRaw$Source = "Test"

# Combine train and test rows back into FullRaw
FullRaw = rbind(TrainRaw, TestRaw)

# % Split of 0s and 1s
table(FullRaw[FullRaw$Source == "Train", "Default_Payment"])/ nrow(FullRaw[FullRaw$Source == "Train",]) # FullRaw
table(TrainRaw$Default_Payment)/nrow(TrainRaw) # TrainRaw

# Summarize the data
summary(FullRaw)

# Remove/Drop Customer ID
FullRaw$Customer.ID = NULL
View(FullRaw)

############################
# Use data description excel sheet to convert numeric variables to categorical variables
############################

# Categorical variables: Gender, Academic_Qualification, Marital

# Gender
levels(FullRaw$Gender) # levels will work on "factor" variables and NOT on "int" variable
FullRaw$Gender = as.factor(FullRaw$Gender)
levels(FullRaw$Gender)



levels(FullRaw$Gender) = c("Male", "Female") # Conversion step
levels(FullRaw$Gender)

# Academic_Qualification

FullRaw$Academic_Qualification = as.factor(FullRaw$Academic_Qualification)
levels(FullRaw$Academic_Qualification)
levels(FullRaw$Academic_Qualification) = c("Undergraduate", "Graduate", "Postgraduate", "Professional", "Others", "Unknown")
levels(FullRaw$Academic_Qualification)

# Marital

FullRaw$Marital = as.factor(FullRaw$Marital)
levels(FullRaw$Marital)
levels(FullRaw$Marital) = c("Unknown", "Married", "Single", "Unknown")
levels(FullRaw$Marital)


############################
# Dummy variable creation
############################

View(FullRaw)

factorVars = sapply(FullRaw, is.factor)
factorVars
colnames(dummyDf)
dummyDf = model.matrix(~ ., data = FullRaw[,factorVars])
View(dummyDf)
FullRaw2 = cbind(FullRaw[,!factorVars], dummyDf[,-1])

View(FullRaw2)

############################
# Divide the data into Train and Test
############################
# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column

Train = subset(FullRaw2, subset = (Source == "Train"), select = -Source)
Test = subset(FullRaw2, subset = (Source == "Test"), select = -Source)

dim(Train)
dim(Test)

############################
# Multicollinearity check
############################
# All VIF should be less than 10 
# (In linear regression: VIF < 5. in logistic regression, its slightly relaxed: VIF < 10)

library(car)
M1 = lm(Default_Payment ~ ., data = Train)
sort(vif(M1), decreasing = TRUE)[1:3]

# Remove MaritalSingle
M2 = lm(Default_Payment ~ . - MaritalSingle, data = Train)
sort(vif(M2), decreasing = TRUE)[1:3]

# Remove May_Bill_Amount
M3 = lm(Default_Payment ~ . - MaritalSingle - May_Bill_Amount, data = Train)
sort(vif(M3), decreasing = TRUE)[1:3]

# Remove March_Bill_Amount
M4 = lm(Default_Payment ~ . - MaritalSingle - May_Bill_Amount - March_Bill_Amount , data = Train)
sort(vif(M4), decreasing = TRUE)[1:3]


# Build logistic regression model
M5 = glm(Default_Payment ~ . - March_Bill_Amount - MaritalSingle - May_Bill_Amount, data = Train, 
         family = "binomial")

summary(M5)
 

############################
# Model optimization (by selecting ONLY significant variables through step() function)
############################

M6 = step(M5)
summary(M6)

# In case you would like to further remove some of the variables based on pvalue, you can use update() function
M7 = update(M6, . ~ . - Academic_QualificationUnknown)
summary(M7)

 M8 = update(M7, . ~ . - Previous_Payment_May)

M9= update(M8,.~.-Age_Years-Repayment_Status_May-Academic_QualificationPostGraduate)
 # summary(M8)


############################
# Prediction and Validation
############################

### Testset Prediction & Validation (with 0.5 Cutoff)

# Predict on Testset using final model
Test_Prob = predict(M9, Test, type = "response") 
# type = "response" will give probability values as output
head(Test_Prob)
head(Test$Default_Payment)

# Classify the test predictions into classes of 0s and 1s
Test_Class = ifelse(Test_Prob >= 0.5, 1, 0)
head(Test_Class)
table(Test_Class)
# Confusion Matrix
table(Test_Class, Test$Default_Payment) # R,C



# Accuracy of the Model
sum(diag(table(Test_Class, Test$Default_Payment)))/nrow(Test) # 81.73%

# TPR = [TP/ Total Actual Positives (1s)]
690/(1333+690) # 34.10%

# FPR = [FP/ Total Actual Negatives (0s)]
311/(6666+311) # 4.45%

### Train Prediction & Validation (with 0.5 Cutoff)

# Predict on Train using final model
Train_Prob = predict(M9, Train, type = "response") # type = "response" gives probability values
head(Train_Prob)
head(Train$Default_Payment)


# Classify the test predictions into classes of 0s and 1s
Train_Class = ifelse(Train_Prob >= 0.5, 1, 0)
head(Train_Class)

# Confusion Matrix
table(Train_Class, Train$Default_Payment)


# Accuracy of the Model
sum(diag(table(Train_Class, Train$Default_Payment)))/nrow(Train) # 81.95%

# TPR = [TP/ Total Actual Positives (1s)]
1523/(1523+3091) # 33.00%

# FPR = [FP/ Total Actual Negatives (0s)]
698/(698+15688) # 4.25%

############################
# ROC Curve
############################


# ROC curve plots the no. of 1s in the model "correctly identified" as 1s (TPR) 
# vs. the no. of 0s in the model "incorrectly identified" as 1 (FPR)
# It helps in finding an alternate probabilty cutoff point

# Confusion Matrix

#                Actual
#                Pos Neg
# Predicted Pos  TP  FP
#           Neg  FN  TN

# TPR = TP/ (TP + FN)
# FPR = TN/ (FP + TN)


# Load ROC library
library(ROCR)

ROC_pred = prediction(Train_Prob, Train$Default_Payment)
# prediction() produces an output which has attributes like 
# True Positives, FP, TN, etc.

ROC_Curve = performance(ROC_pred, "tpr", "fpr")

# performance(): All kinds of predictor evaluations 
#(like tpr, fpr, auc, etc.) are performed using this function.
# It helps in creating the ROC Curve

# ROC Curve 
plot(ROC_Curve)

# Cutoff point selection table: 
# The first column gives the probability values that can be chosen
# as a cutoff point (i.e. something other than 0.5 probability value taken above). 
# The second and third columns are FPR and TPR
# The ideal cutoff would be to maximize TPR and minimize FPR!

Cutoff_Table = cbind.data.frame(Cutoff = ROC_Curve@alpha.values[[1]], # Cutoff
                                FPR = ROC_Curve@x.values[[1]], # FPR
                                TPR = ROC_Curve@y.values[[1]]) # TPR

View(Cutoff_Table)


############

# # AUC Interpretation: Higher the better (Range: 0 to 1)

ROC_AUC = performance(ROC_pred, "auc")
ROC_AUC@y.values # AUC



############################
# Improve Model Output Using New Cutoff Point
############################

Cutoff_Table$Distance = sqrt((1-Cutoff_Table$TPR)^2 + (0-Cutoff_Table$FPR)^2) 

# Euclidean Distance

Cutoff_Table$MaxDiffBetweenTPRFPR = Cutoff_Table$TPR - Cutoff_Table$FPR 

# Max Diff. Bet. TPR & FPR

View(Cutoff_Table)
max(Cutoff_Table$MaxDiffBetweenTPRFPR)

# New Cutoff Point Performance (Obtained after studying ROC Curve 
# and Cutoff Table)
cutoffPoint = 0.2197922 # Max Difference between TPR & FPR

# Classify the test predictions into classes of 0s and 1s
Test_Class = ifelse(Test_Prob >= cutoffPoint, 1, 0)

# Confusion Matrix
table(Test_Class, Test$Default_Payment)

# Accuracy of the Model
sum(diag(table(Test_Class, Test$Default_Payment)))/nrow(Test) # 77.37%


# TPR
1168/(1168+855) # 57.73%

# FPR
1181/(1181+5796) # 16.9%


############################
# Model Assesment
############################

# All the below output is on "TEST" dataset

# Original model (based on 0.5 cutoff) gave us the following output on Test:
# Accuracy: 81.84%
# TPR: 34.01%
# FPR: 4.11%

# Model evaluation with cutoffPoint = 0.2197922, gave us the following output on Test:
# Accuracy: 77.37%
# TPR: 57.73%
# FPR: 16.9%

# Final model selction depends a lot on what business/ client is looking for.
# Here, although model 1 has a higher accuracy(81.84%), model 2 has gone up by close to 24%
# TPR (57.73%) with the accuracy just dropping by 4%. Since, in loan defaulting cases, 
# higher TPR is very important, one would select model 2 as the final model
