#Set working directory
setwd("F:/Academic/MIT_year 3_Sem1/Big Data/Kaggle_Titanic")
 
#Assign train.csv data to Titanic,train model 
Titanic.train <- read.csv(file = "train.csv", stringsAsFactors = FALSE, header = TRUE)
#Assign test.csv data to Titanic,test model
Titanic.test <- read.csv(file = "test.csv", stringsAsFactors = FALSE, header = TRUE)

#Set True in TrainSet column
Titanic.train$IsTrainSet <- TRUE
#Set False in TrainSet column
Titanic.test$IsTrainSet <- FALSE

#Survives column set not available
Titanic.test$Survived <- NA

#Combine Titanic.train and Titanic.test
Titanic.full <- rbind(Titanic.train, Titanic.test)

#Select "Embarked" value and replace with 'S'
Titanic.full[Titanic.full$Embarked=='', "Embarked"] <- 'S'

#Search median and assign to age.median variable
age.median <- median(Titanic.full$Age, na.rm = TRUE)

#Assign median age value to age column
Titanic.full[is.na(Titanic.full$Age), "Age"] <- age.median

#Set upper  bound
upper.whisker <- boxplot.stats(Titanic.full$Fare)$stats[5]
outlier.filter <- Titanic.full$Fare < upper.whisker
#Filter all according to the limits
Titanic.full[outlier.filter,]

#Prepare Predicting equation
fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(
  formula = fare.equation,
  data = Titanic.full[outlier.filter,]
)

#Prediction query
fare.row <- Titanic.full[
  is.na(Titanic.full$Fare),
  c("Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked")
]

#Predict according to the fare
fare.predictions <- predict(fare.model, newdata = fare.row )
Titanic.full[is.na(Titanic.full$Fare), "Fare"] <- fare.predictions

#categorical casting
Titanic.full$Pclass <- as.factor(Titanic.full$Pclass)
Titanic.full$Sex<- as.factor(Titanic.full$Sex)
Titanic.full$Embarked<- as.factor(Titanic.full$Embarked)

#split dataset back out into train and test
Titanic.train<-Titanic.full[Titanic.full$IsTrainSet==TRUE,] 
Titanic.test<-Titanic.full[Titanic.full$IsTrainSet==FALSE,] 

#Cast the survived into Survived category
Titanic.train$Survived <- as.factor(Titanic.train$Survived)

#Create survived ewuation
survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.formula <- as.formula(survived.equation)

#Install and import "randomForest"
install.packages("randomForest")
library(randomForest)

#Create predictive model using "randomForest" library
Titanic.model <- randomForest(formula = survived.formula, data = Titanic.train, ntree = 500, mtry = 3, nodesize =0.01 * nrow(Titanic.test)) 

#Create feature equation
features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
Survived <- predict(Titanic.model, newdata = Titanic.test)

#Isolate passengerID
PassengerId <- Titanic.test$PassengerId
#Convert into dataframe and assign to output.df
output.df <- as.data.frame(PassengerId)
#Through Survived into output.df
output.df$Survived <- Survived

#Get output prediction into "Submission.csv" 
write.csv(output.df, file="Submission.csv", row.names = FALSE)
