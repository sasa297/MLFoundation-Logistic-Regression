############ Saikiran N. Pasikanti #####################

##1(a)#######################################################
# a) import the Loan-Approval-Prediction.csv
Data1 <- read.csv("C:\\001_SAIKIRAN\\AEGIS\\ML\\Assignment 2\\Loan-Approval-Prediction.csv",
                  header=T, na.strings = c("","NaN"," ",NA))
Data2 <- Data1 #backup

dim(Data1) # 4816 observations, 12 features/variables
summary(Data1) # summary of all variables
# From Summary it is clear that yes to no ratio is 422/199 which is biased
str(Data1)
#plot(Data1)


##1(b)#####################################################
# b) Use Appropriate Imputation Techniques to Predict Missing Values.

library(ggplot2)
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Credit_History))

#Credit History is a high impact variable
Data1$Credit_History = as.character(Data1$Credit_History)
Data1$Credit_History[is.na(Data1$Credit_History)] = "Not Available"
Data1$Credit_History = as.factor(Data1$Credit_History)

print(ggplot(Data1, aes(x=Loan_Status,y=ApplicantIncome))+geom_boxplot())
print(ggplot(Data1, aes(x=Loan_Status,y=CoapplicantIncome))+geom_boxplot())
print(ggplot(Data1, aes(x=Loan_Status,y=LoanAmount))+geom_boxplot())

#Feature Engineering
Data1$TotalIncome <- log(Data1$ApplicantIncome + Data1$CoapplicantIncome)
Data1$TotalIncomeLoanRatio = log(((Data1$ApplicantIncome + Data1$CoapplicantIncome)/Data1$LoanAmount)*(as.numeric(Data1$Loan_Amount_Term)/360))
Data1 <- Data1[,!(names(Data1)) %in% c("ApplicantIncome","CoapplicantIncome")]

#imputing missing loan amount using sub categories
print(boxplot(Data1$LoanAmount)$stats)
ind <- which(is.na(Data1$LoanAmount))
Data1[ind,]$LoanAmount[Data1[ind,]$Education == "Graduate" & Data1[ind,]$Self_Employed == "No"] <- 145.82
Data1[ind,]$LoanAmount[Data1[ind,]$Education == "Graduate" & Data1[ind,]$Self_Employed == "Yes"] <- 174.24
Data1[ind,]$LoanAmount[Data1[ind,]$Education == "Not Graduate" & Data1[ind,]$Self_Employed == "No"] <- 116.7
Data1[ind,]$LoanAmount[Data1[ind,]$Education == "Not Graduate" & Data1[ind,]$Self_Employed == "Yes"] <- 131.56

View(Data1)

# Checking the impact of other variables on Loan Status
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Gender))
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Married))
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Dependents))
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Education))
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Self_Employed))
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Loan_Amount_Term))
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Credit_History))
print(ggplot(Data1, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Property_Area))

# replacing the missing values of these variables
#install.packages("mice")
library(mice)
init = mice(Data1, maxit=0) 
meth = init$method
predM = init$predictorMatrix

predM[,c("Loan_ID")]=0

meth[c("Gender")]="logreg"
meth[c("Married")]="logreg"
meth[c("Dependents")]="polyreg"
meth[c("Self_Employed")]="logreg"
meth[c("Loan_Amount_Term")]="norm"

set.seed(111)
Data.Imp = mice(Data1, method=meth, predictorMatrix=predM, m=5)
Data.Imp <- complete(Data.Imp)
sapply(Data.Imp, function(x) sum(is.na(x)))
# All the missing values are imputed successfully

##1(c)########################################################
# c) Divide the data into training and testing set in 70:30 ratio.

sample <- floor(0.7 * nrow(Data.Imp))
set.seed(111)
train_ind <- sample(seq_len(nrow(Data.Imp)), size = sample)

Data.train <- Data.Imp[train_ind, ]
Data.test <- Data.Imp[-train_ind, ]

row.names(Data.train) <- 1:nrow(Data.train)


##1(d)###########################################################
# d) Draw Information Value Summary plot.

library(woe)

iv.plot.summary(iv.mult(Data.train[,!names(Data.train) %in% c("Loan_ID")], "Loan_Status",TRUE))

iv <- iv.mult(Data.train[,!names(Data.train) %in% c("Loan_ID")], "Loan_Status",TRUE)

iv


##1(e)###########################################################
# e) Build the logistic model for checking Loan_Status with important features which we got from IV plot.
model1 <- glm(Loan_Status ~  Credit_History + TotalIncomeLoanRatio + TotalIncome + LoanAmount + Property_Area,
              data = Data.train, family = "binomial")
summary(model1)

model2 <- glm(Loan_Status ~  Credit_History + Property_Area,data = Data.train, family = "binomial")
summary(model2)

fitted.probabilities <- predict(model2, newdata=Data.test, type='response')
table(Data.test$Loan_Status, fitted.probabilities > 0.5)


##1(f)############################################################
# f) find out the optimum cutoff probability to reduce the missclassification error.
library(InformationValue)
OCP <- optimalCutoff(Data.test$Loan_Status, fitted.probabilities)[1] 
OCP

fitted.probabilities <- predict(model2,newdata=Data.test,type='response')
table(Data.test$Loan_Status, fitted.probabilities > 0.04078136)
