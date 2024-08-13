if (!require(e1071)) install.packages("dplyr")
if (!require(e1071)) install.packages("tidyverse")
if (!require(e1071)) install.packages("caret")
if (!require(e1071)) install.packages("ggplot2")
if (!require(e1071)) install.packages("rpart")
if (!require(e1071)) install.packages("rpart.plot")
if (!require(e1071)) install.packages("randomForest")
if (!require(e1071)) install.packages("e1071")

library("dplyr")
library("tidyverse")
library("caret")
library("ggplot2")
library("rpart")
library("rpart.plot")
library("randomForest")
library("e1071")

# The data set used for this project can be found at:
# https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

# Introduction

#This is the last graded project of the Capstone course of the Harvard Data Science series. 
#For this project, I have chosen a data set to practice my machine learning knowledge. 
#The data set that I will be using is the "Human Activity Recognition Using Smartphones" (HARUS) data set. 
#The HARUS data set consists of the recordings from 30 participants, 
#who performed daily tasks while wearing a smartphone around their waist. 

#The participants were aged between 19 and 48 years of age and each participant performed 6 daily activities. 
#These activities were: walking, walking downstairs, sitting, standing, and laying. 
#The sampling rate of the recordings was 50Hz and stored as time series per dimension. 
#So, six different signals were measured: 3 from the accelerometer and 3 from the gyroscope.  

#The data set has already been divided into training and test sets. 
#Of the participants, 70% was randomly selected to form the training set, 
#and 30% was selected to form the test set. 

#In this project, I will develop a model that determines the type of activity, 
#based on data collected by smartphone sensors. 


# Loading and preparing the data

# Loading the train data
X_train <- read.table("X_train.txt")
y_train <- read.table("y_train.txt")
subject_train <- read.table("subject_train.txt")

# Loading the test data
X_test <- read.table("X_test.txt")
y_test <- read.table("y_test.txt")
subject_test <- read.table("subject_test.txt")

# Loading features and activity labels
features <- read.table("features.txt")
feature_names <- as.character(features$V2)
activity_labels <- read.table("activity_labels.txt")

# Assigning an ID and name to activities
colnames(activity_labels) <- c("ActivityID", "ActivityName")

# Assigning feature names to the X variables
colnames(X_train) <- feature_names
colnames(X_test) <- feature_names

# Assigning column names to the y and subject data
colnames(y_train) <- "Activity"
colnames(subject_train) <- "Subject"
colnames(y_test) <- "Activity"
colnames(subject_test) <- "Subject"

# Combining train data into one
train_data <- cbind(subject_train, y_train, X_train)

# Combining test data into one
test_data <- cbind(subject_test, y_test, X_test)

# Checking for duplicate columns and rename if necessary
colnames(train_data) <- make.names(colnames(train_data), unique = TRUE)
colnames(test_data) <- make.names(colnames(test_data), unique = TRUE)

# Transforming activity into a categorical variable
train_data$Activity <- factor(train_data$Activity, levels = activity_labels$ActivityID, labels = activity_labels$ActivityName)
test_data$Activity <- factor(test_data$Activity, levels = activity_labels$ActivityID, labels = activity_labels$ActivityName)

# Combining all data into one data frame for data exploration
combined_data <- rbind(
  cbind(subject_train, y_train, X_train),
  cbind(subject_test, y_test, X_test)
)

# Before we start the data exploration, let's check for missing values

# Checking for missing values
sum(is.na(combined_data))


# Data Exploration

# Distribution of activities
table(combined_data$Activity)

# Checking for duplicate column names
duplicate_columns <- duplicated(colnames(combined_data))
print(duplicate_columns)

# Showing duplicate column names
duplicate_column_names <- colnames(combined_data)[duplicate_columns]
duplicate_column_names

# Renaming duplicates
colnames(combined_data) <- make.names(colnames(combined_data), unique = TRUE)


# Ensuring the 'Activity' column is a factor with correct labels
combined_data$Activity <- factor(combined_data$Activity, 
                                 levels = c(1, 2, 3, 4, 5, 6), 
                                 labels = c("Walking", "Walking Upstairs", "Walking Downstairs", 
                                            "Sitting", "Standing", "Laying"))

# Bar plot of activity distribution
ggplot(combined_data, aes(x = Activity)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Activity Distribution", x = "Activity", y = "Count")

# Methods

# To predict the right activity based on the smartphone sensor data, 
# I will use several models and test the accuracy of each. 
# First, I will develop a baseline model to compare the results to. 

# Baseline model
# The baseline model I will use, is a decision tree. 
# A decision tree model splits the data its given into smaller sets at each decision node.
# This way, what's left in the end is hopefully the type of activity that corresponds to the data. 

# Decision tree

# Fitting the decision tree model
tree_model <- rpart(Activity ~ ., data = train_data, method = "class")

# Plotting the decision tree
rpart.plot(tree_model, type = 3, extra = 101, fallen.leaves = TRUE, main = "Decision Tree for Activity Recognition")

# Predicting on the test set
tree_predictions <- predict(tree_model, test_data, type = "class")

# Calculating the confusion matrix and accuracy
tree_conf_matrix <- confusionMatrix(tree_predictions, test_data$Activity)

# Showing the confusion matrix
tree_conf_matrix

# Calculating accuracy
tree_accuracy <- as.numeric(tree_conf_matrix$overall['Accuracy'])

# Showing a table with the model and its corresponding accuracy
accuracy_table <- data.frame(Model = "Decision Tree", Accuracy = tree_accuracy)
accuracy_table

# The first model I will test and compare to the baseline model, is random forest. 
# Random forests is a model that is made up of several decision trees. 
# In this case, the variable needs to be classified. 
# I.E., the type of activity is determined. 
# So, in the end the most frequent activity type over all decision trees, will be predicted. 


# Random Forest

set.seed(123)  # For reproducibility

# Training the Random Forest model
rf_model <- randomForest(Activity ~ ., data = train_data, ntree = 100, importance = TRUE)

rf_model

# Using the model to predict based on the test set
rf_predictions <- predict(rf_model, test_data)

# Confusion matrix and accuracy
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Activity)
rf_conf_matrix

# Showing the overall accuracy
rf_accuracy <- cat("Accuracy: ", rf_conf_matrix$overall['Accuracy'], "\n")
rf_accuracy <- as.numeric(rf_conf_matrix$overall['Accuracy'])

# Showing a table with accuracies from both the decision tree and RF models
accuracy_table <- rbind(accuracy_table, data.frame(Model = "Random Forest", Accuracy = rf_accuracy))
accuracy_table

# Support Vector Machines

# The next and last model we will test is the Support Vector Machines (SVM) model.
# SVM tries to find a hyperplane that separates the points of different classes.
# SVM then tries to find the hyperplane that has the largest distance between the points from the different classes. 

# Setting seed for reproducibility
set.seed(123)

# Training the SVM model with a linear kernel
svm_model <- svm(Activity ~ ., data = train_data, kernel = "linear", scale = TRUE)

svm_model

# Predicting on the test set
svm_predictions <- predict(svm_model, test_data)

# Confusion matrix and accuracy
svm_conf_matrix <- confusionMatrix(svm_predictions, test_data$Activity)
svm_conf_matrix

# Showing the overall accuracy
svm_accuracy <- cat("Accuracy: ", svm_conf_matrix$overall['Accuracy'], "\n")
svm_accuracy <- as.numeric(svm_conf_matrix$overall['Accuracy'])

# Showing a table with accuracies from all models
accuracy_table <- rbind(accuracy_table, data.frame(Model = "SVM", Accuracy = svm_accuracy))
accuracy_table

# Results

accuracy_table

# Conclusion

# In this project I compared two models, Random Forest and SVM, to the baseline model, decision tree. 
# I used these models to predict activity type, based on data collected by smartphone sensors. 

# The baseline model already worked quite well, with an accuracy of 0.840.
# The RF model was a good improvement with an accuracy of 0.928. 
# And the SVM model worked best, with an accuracy of 0.957.

# This project has some limitations.
# First of all, the data was recorded from only 30 participants. 
# This could cause, that, when these models would be used on "real world" data,
# It could be less precise in predicting activity type correctly. 
# Furthermore, the activity types were recorded in a controlled setting. 
# Again, when "real world" data would be used, It could be quite noisy and hard to predict. 
# Lastly, although RF and SVM seem like the right models, 
# these could still have overfitted the data. 
# This would mean, that although these models seem to work good on the test data, 
# they might not on real data. 

# Finally, for the future, it would be interesting to see how these models would do with "real world" data.

# References

# Reyes-Ortiz,Jorge, Anguita,Davide, Ghio,Alessandro, Oneto,Luca, and Parra,Xavier. (2012). Human Activity Recognition Using Smartphones. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K.

# R.A. Irizarry. (2019). Introduction to Data Science.

# IBM. "What Is Random Forest?" IBM, https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems. Accessed 13 Aug. 2024.

# GeeksforGeeks. "Support Vector Machine Algorithm." GeeksforGeeks, https://www.geeksforgeeks.org/support-vector-machine-algorithm/. Accessed 13 Aug. 2024.