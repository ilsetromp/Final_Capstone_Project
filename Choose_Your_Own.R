library("dplyr")
library("tidyverse")
library("ggplot2")


# Human Activity Recognition Using Smartphones data set
# The data is already split into training and test sets


# Loading the data

# Loading the training data
X_train <- read.table("X_train.txt")
y_train <- read.table("y_train.txt")
subject_train <- read.table("subject_train.txt")

# Loading the test data
X_test <- read.table("X_test.txt")
y_test <- read.table("y_test.txt")
subject_test <- read.table("subject_test.txt")

# Loading feature names
features <- read.table("features.txt")
feature_names <- as.character(features$V2)

# Assigning column names to the training and test sets
colnames(X_train) <- feature_names
colnames(X_test) <- feature_names

# We will combine the training and test sets in order to do exploratory data analysis

# Combining the training and test sets
X_data <- rbind(X_train, X_test)
y_data <- rbind(y_train, y_test)
subject_data <- rbind(subject_train, subject_test)

# Assigning names to the y_data and subject_data columns
colnames(y_data) <- "Activity"
colnames(subject_data) <- "Subject"

# Combining all data into one data frame
combined_data <- cbind(subject_data, y_data, X_data)

# The activities are encrypted by numbers
# We will assign corresponding activities

# Loading activity labels
activity_labels <- read.table("activity_labels.txt")
colnames(activity_labels) <- c("ActivityID", "ActivityName")

# Replace activity numbers with descriptive names
combined_data$Activity <- factor(combined_data$Activity, levels = activity_labels$ActivityID, labels = activity_labels$ActivityName)


# Before we start the data exploration, let's check for missing values

# Checking for missing values
sum(is.na(combined_data))
