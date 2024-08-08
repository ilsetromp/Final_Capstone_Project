library("dplyr")
library("tidyverse")
library("caret")
library("ggplot2")
library("rpart")
library("rpart.plot")
library("e1071")
library("randomForest")

# Introduction

#This is the last graded project of the Capstone course of the Harvard Data Science series. 
#For this project, I have chosen a data set to practice my machine learning knowledge. 
#The data set that I will be using is the "Human Activity Recognition Using Smartphones" (HARUS) data set. 
#The HARUS data set consists of the recordings from 30 participants, 
#who performed daily tasks while wearing a smartphone around their waist. 

#The participants were aged between 19 and 48 years of age and each participant performed 6 daily activities. 
#These activities were: walking, walking downstairs, sitting, standing, and laying. 
#The sampling rate of the recordings was 50Hz and stored as time series per dimension. 
#So, six different signals were measured: 3 from the accerlerometer and 3 from the gyroscope.  

#The data set has already been divided into training and test sets. 
#Of the participants, 70% was randomly selected to form the training set, 
#and 30% was selected to form the test set. 

#In this project, I will develop a model that determines the type of activity, 
#based on data collected by smartphone sensors. 

# Loading the data

# Load and prepare the data
X_train <- read.table("X_train.txt")
y_train <- read.table("y_train.txt")
subject_train <- read.table("subject_train.txt")

X_test <- read.table("X_test.txt")
y_test <- read.table("y_test.txt")
subject_test <- read.table("subject_test.txt")

features <- read.table("features.txt")
feature_names <- as.character(features$V2)
activity_labels <- read.table("activity_labels.txt")
colnames(activity_labels) <- c("ActivityID", "ActivityName")

colnames(X_train) <- feature_names
colnames(X_test) <- feature_names

colnames(y_train) <- "Activity"
colnames(subject_train) <- "Subject"
colnames(y_test) <- "Activity"
colnames(subject_test) <- "Subject"

train_data <- cbind(subject_train, y_train, X_train)
test_data <- cbind(subject_test, y_test, X_test)

# Check for duplicate columns and rename if necessary
colnames(train_data) <- make.names(colnames(train_data), unique = TRUE)
colnames(test_data) <- make.names(colnames(test_data), unique = TRUE)

train_data$Activity <- factor(train_data$Activity, levels = activity_labels$ActivityID, labels = activity_labels$ActivityName)
test_data$Activity <- factor(test_data$Activity, levels = activity_labels$ActivityID, labels = activity_labels$ActivityName)

# Combining all data into one data frame
combined_data <- cbind(subject_data, y_data, X_data)

# Before we start the data exploration, let's check for missing values

# Checking for missing values
sum(is.na(combined_data))


# Data Exploration

# Distribution of activities
table(combined_data$Activity)

# Check for duplicate column names
duplicate_columns <- duplicated(colnames(combined_data))
print(duplicate_columns)

# Print the names of duplicate columns
duplicate_column_names <- colnames(combined_data)[duplicate_columns]
print(duplicate_column_names)

# Renaming duplicates
colnames(combined_data) <- make.names(colnames(combined_data), unique = TRUE)

# Plot the data again
# Bar plot of activity distribution
ggplot(combined_data, aes(x = Activity)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Activity Distribution", x = "Activity", y = "Count")

# Boxplot of a sample feature for each activity
ggplot(combined_data, aes(x = Activity, y = combined_data[, 3])) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of Feature 1 by Activity", x = "Activity", y = colnames(combined_data)[3])


# Model development

# Baseline model
# Decistion tree

# Fitting the decision tree model
tree_model <- rpart(Activity ~ ., data = train_data, method = "class")

# Plotting the decision tree
rpart.plot(tree_model, type = 3, extra = 101, fallen.leaves = TRUE, main = "Decision Tree for Activity Recognition")

# Predicting on the test set
tree_predictions <- predict(tree_model, test_data, type = "class")

# Calculating the confusion matrix and accuracy
tree_conf_matrix <- confusionMatrix(tree_predictions, test_data$Activity)

# Showing the confusion matrix
print(tree_conf_matrix)

# Random Forests
# Train the Random Forest model
set.seed(123)  # For reproducibility
rf_model <- randomForest(Activity ~ ., data = train_data, ntree = 100, importance = TRUE)

rf_model

# Using the model to predict based on the test set
rf_predictions <- predict(rf_model, test_data)

# Confusion matrix and accuracy
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$Activity)
print(rf_conf_matrix)

# Print the overall accuracy
cat("Accuracy: ", rf_conf_matrix$overall['Accuracy'], "\n")



# Support Vector Machines
