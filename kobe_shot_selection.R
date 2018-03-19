# Get original data
original_kobe <- read.csv("kobe_data.csv")
train <- read.csv("kobe_data.csv") # train data
options(scipen=999)

# Function : normalize data into range 0:1
normalize <- function(data) {
  data <- (data - min(data)) / (max(data) - min(data))
}
standardize <- function(data) {
  data <- (data - mean(data)) / sd(data)
}

# Convert date to weekday and other
# install.packages("lubridate") #remove comment here if you didn't install this package yet
library(lubridate)
train$week_day <- wday(train$game_date)
train$month <- month(train$game_date)
train$year <- year(train$game_date)

# Normalize data
train$loc_x <- normalize(train$loc_x)
train$loc_y <- normalize(train$loc_y)
train$shot_distance <- normalize(train$shot_distance)
train$time_remaining <- (train$minutes_remaining*60) + train$seconds_remaining
train$time_remaining <- normalize(train$time_remaining)
train$period <- normalize(train$period)
train$matchup <- ifelse( grepl("@",train$matchup), train$matchup <- "AWAY", train$matchup <- "HOME" ) 
train$week_day <- normalize(train$week_day)
train$month <- normalize(train$month)
train$year <- normalize(train$year)

# Remove unused attributes (Useless for data modeling such as _id)
train$seconds_remaining <- NULL  # use time_remaining instead
train$minutes_remaining <- NULL  # use time_remaining instead
train$game_id <- NULL
train$game_event_id <- NULL
train$game_date <- NULL         # use weekday,month,year
train$team_id <- NULL
train$team_name <- NULL         # Kobe NEVER play for other teams
train$season <- NULL
train$shot_id <- NULL
train$lat <- NULL   
train$lon <- NULL     

# Prepare for model
#install.packages("dummies") #remove comment here if you didn't install this package yet
library(dummies)
df<- dummy.data.frame(train , names = c('action_type','combined_shot_type' , 'shot_type' , 'shot_zone_area' ,'playoffs' , 'shot_zone_basic' , 'shot_zone_range' , 'matchup' , 'opponent') , sep='_')
X_train <- df[(!is.na(df$shot_made_flag)),]
X_test <- df[(is.na(df$shot_made_flag)),]
y <- X_train$shot_made_flag

set.seed(100) 
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(X_train), size = floor(.75*nrow(X_train)), replace = F)
X_train_sample <- X_train[sample, ]
X_test_sample  <- X_train[-sample, ]
y_sample <- X_train_sample$shot_made_flag
X_test_sample2 <- X_test_sample
X_train_sample$shot_made_flag <- NULL # required by label xgboost that is shouldn't be provided in the data
X_test_sample$shot_made_flag <- NULL  # required by label xgboost that is shouldn't be provided in the data
#install.packages("xgboost") #remove comment here if you didn't install this package yet
library(xgboost)  # decision tree with strong learners for ML
# xgboost model <Based on decision tree>
model_xgb_sample <- xgboost(data=as.matrix(X_train_sample), label=as.matrix(y_sample), objective="binary:logistic",nrounds=200, learning_rate=0.02, max_depth=8, subsample=0.6, colsample_bytree=0.85, min_child_weight=1, eval_metric="logloss")
pred_sample <- predict(model_xgb_sample, as.matrix(X_test_sample))
# SAMPLE TEST
original_not_na <- na.omit(original_kobe)
true_result <- as.numeric(X_test_sample2$shot_made_flag)
error <- abs(as.numeric(X_test_sample2$shot_made_flag)-pred_sample)
true_pos <- vector(mode="numeric", length=6425)
false_pos <- vector(mode="numeric", length=6425)
true_neg <- vector(mode="numeric", length=6425)
false_neg <- vector(mode="numeric", length=6425)
for (i in 1:6425) {
  true_pos[i] <- ifelse(true_result[i]==1&&pred_sample[i]>=0.5, true_pos[i]<-1,true_pos[i]<-0)
  false_pos[i] <- ifelse(true_result[i]==0&&pred_sample[i]>=0.5, false_pos[i]<-1,false_pos[i]<-0)
  true_neg[i] <- ifelse(true_result[i]==0&&pred_sample[i]<0.5, true_neg[i]<-1,true_neg[i]<-0)
  false_neg[i] <- ifelse(true_result[i]==1&&pred_sample[i]<0.5, false_neg[i]<-1,false_neg[i]<-0)
}
accuracy <- (sum(true_neg) + sum(true_pos)) / (sum(true_pos) + sum(false_pos) + sum(true_neg) + sum(false_neg))
recall <- sum(true_pos) / (sum(true_pos) + sum(false_neg))
precision <- sum(true_pos) / (sum(true_pos) + sum(false_pos))
model_F_predict_F <- sum(true_neg) / (sum(true_neg) + sum(false_neg))
result_sample <- data.frame("shot_id"=as.numeric(rownames(X_test_sample2)), "TRUE_RESULT"= true_result, "PREDICTION"=pred_sample,"TRUE_POS" = sum(true_pos),"FALSE_POS"=sum(false_pos),"TRUE_NEG"=sum(true_neg),"FALSE_NEG"=sum(false_neg),
                              "ACC" = accuracy, "PRECISION" = precision, "RECALL" = recall)
write.csv(result_sample , '~/Desktop/kobe_prediction_result_sample3.csv')

X_train$shot_made_flag <- NULL # required by label xgboost that is shouldn't be provided in the data
X_test$shot_made_flag <- NULL  # required by label xgboost that is shouldn't be provided in the data
model_xgb <- xgboost(data=as.matrix(X_train), label=as.matrix(y), objective="binary:logistic", nrounds=200, learning_rate=0.02, max_depth=8, subsample=0.6, colsample_bytree=0.85, min_child_weight=1, eval_metric="logloss")
pred <- predict(model_xgb, as.matrix(X_test))

# Construct result
result <- data.frame("shot_id"=as.numeric(rownames(X_test)), "shot_made_flag"=pred)
write.csv(result , '~/Desktop/kobe_prediction_result.csv', row.names = F)
