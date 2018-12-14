#Loading Packages
library(tidyverse)
library(jsonlite)
library(magrittr)
library(stringr)
library(caret)
library(lubridate)
library(data.table)
library(scales)
library(ggplot2)
library(lightgbm)
library(doParallel)
library(corrplot)
library(rpart.plot)
library(rsample)
library(rattle)
library(keras)
#Defining Data Types in File for Faster Loading of 27GB file
ctypes <- cols(fullVisitorId = col_character(),
               channelGrouping = col_character(),
               date = col_datetime(),
               device = col_character(),
               geoNetwork = col_character(),
               socialEngagementType = col_skip(),
               totals = col_character(),
               trafficSource = col_character(),
               visitId = col_integer(),
               visitNumber = col_integer(),
               visitStartTime = col_integer(),
               hits = col_skip(),
               customDimensions = col_skip())
#Loading Dataset
tr <- read_csv("train_v2.csv", col_types = ctypes).
#Functions to Flatten Json Fields
flatten_json <- . %>%
  str_c(., collapse = ",") %>%
  str_c("[", ., "]") %>%
  fromJSON(flatten = T)
parse <- . %>%
  bind_cols(flatten_json(.$device)) %>%
  bind_cols(flatten_json(.$geoNetwork)) %>%
  bind_cols(flatten_json(.$trafficSource)) %>%
  bind_cols(flatten_json(.$totals)) %>%
  select(-device, -geoNetwork, -trafficSource, -totals)
#Parse Dataset to Flatten Json to Table Columns
tr <- parse(tr)
#Seperate Out Response Variable from Train Set
tr_te$transactionRevenue <- as.numeric(tr$transactionRevenue)
y <- log1p(as.numeric(tr$transactionRevenue))
y[is.na(y)] <- 0
tr$transactionRevenue <- NULL
tr$campaignCode <- NULL
#Auxilarry Functions for Processing
has_many_values <- function(x) n_distinct(x) > 1
#Defining all Values that should be considered missing data
is_na_val <- function(x) x %in% c("not available in demo dataset", "(not provided)",
                                  "(not set)", "<NA>", "unknown.unknown", "(none)")
#Data Preprocessing
tr_te <- tr %>%
  select_if(has_many_values) %>%
  mutate_all(funs(ifelse(is_na_val(.), NA, .))) %>%
  mutate(pageviews = ifelse(is.na(pageviews), 0L, as.integer(pageviews)),
         visitNumber = visitNumber,
         newVisits = ifelse(newVisits == "1", 1L, 0L),
         bounces = ifelse(is.na(bounces), 0L, 1L),
         isMobile = ifelse(isMobile, 1L, 0L),
         adwordsClickInfo.isVideoAd = ifelse(is.na(adwordsClickInfo.isVideoAd), 0L, 1L),
         isTrueDirect = ifelse(is.na(isTrueDirect), 0L, 1L),
         browser_dev = str_c(browser, "_", deviceCategory),
         browser_os = str_c(browser, "_", operatingSystem),
         browser_chan = str_c(browser, "_", channelGrouping),
         campaign_medium = str_c(campaign, "_", medium),
         chan_os = str_c(operatingSystem, "_", channelGrouping),
         country_adcontent = str_c(country, "_", adContent),
         country_medium = str_c(country, "_", medium),
         country_source = str_c(country, "_", source),
         dev_chan = str_c(deviceCategory, "_", channelGrouping),
         date = as_datetime(visitStartTime),
         year = year(date),
         wday = wday(date),
         hour = hour(date))
#Feature Engineering - New Combination Features
for (i in c("city", "country", "networkDomain"))
  for (j in c("browser", "operatingSystem", "source"))
    tr_te[str_c(i, "_", j)] <- str_c(tr_te[[i]], tr_te[[j]], sep = "_")
#Feature Engineering - New User Group Features Ex.No of Unique Users per Day,Hr
for (grp in c("wday", "hour")) {
  col <- paste0(grp, "_user_cnt")
  tr_te %<>%
    group_by_(grp) %>%
    mutate(!!col := n_distinct(fullVisitorId)) %>%
    ungroup()
}
#Feature Engineering - Aggregate Functions on Network Domain, Referral Path and Number of Visits
fn <- funs(mean, min, max, sum, .args = list(na.rm = TRUE))
for (grp in c("networkDomain", "referralPath", "visitNumber")) {
  df <- paste0("sum_by_", grp)
  s <- paste0("_", grp)
  tr_te %<>%
    left_join(assign(df, tr_te %>%
                       select_(grp, "pageviews") %>%
                       group_by_(grp) %>%
                       summarise_all(fn)), by = grp, suffix = c("", s))
}
#Final Steps in Preprocessing
tr_te %<>%
  select(-date,-dev_chan,-fullVisitorId, -visitId, -visitStartTime, -hits, -totalTransactionRevenue) %>%
  mutate_if(is.character, funs(factor(.) %>% addNA %>% as.integer)) %>%
  mutate_if(is.integer, funs(as.numeric)) %>%
  select_if(has_many_values)
#Calculate Near Zero Variance at 95%
nzv <- nearZeroVar(tr_te)
#Removing Columns with near zero variance
tr_te <- tr_te[,-nzv]
glimpse(tr_te)
#Create Partition and Folds
part_ind <- createDataPartition(y,p=0.7,list=FALSE)
train.data <- tr_te[part_ind,]
test.data <- tr_te[-part_ind,]
train.output <- y[part_ind]
test.output <- y[-part_ind]
folds_5=createFolds(y=train.output,k=5,returnTrain=TRUE)
#Starting Model Training
print("==== Linear Model========")
linear_model <- train(y=train.output,
                      x=as.matrix(train.data),
                      metric="RMSE",
                      method = "lm",
                      trControl = trainControl(method="cv",number=5,index=folds_5)
)
linear_model$resample
saveRDS(linear_model,"linear_model.rds")
res_linear_model <- predict(linear_model,as.matrix(test.data))
postResample(res_linear_model,test.output)
print("======Decision Tree===========")
dtree_model1 <- train(y=train.output,
                      x=as.matrix(train.data),
                      metric="RMSE",
                      method = "rpart",
                      trControl = trainControl(method="cv",number=5,index=folds_5)
)
saveRDS(dtree_model,"dtree_model.rds")
res_dtree_model <- predict(dtree_model1,as.matrix(test.data))
RMSE(res_dtree_model,test.output)
fancyRpartPlot(dtree_model1$finalModel)
print("======Bagging===========")
bag_model <- train(y=train.output,
                   x=as.matrix(train.data),
                   metric="RMSE",
                   method = "treebag",
                   ntree = 5,
                   trControl = trainControl(method="cv",number=5, index=folds_5)
)
saveRDS(bag_model,"bag_model.rds")
res_bag_model <- predict(bag_model,as.matrix(test.data))
RMSE(res_bag_model,test.output)
print("======Random Forest in Parallel===========")
rf_model <- train(y=train.output,
                  x=as.matrix(train.data),
                  metric="RMSE",
                  method = "rf",
                  ntree=32,
                  nthread=32,
                  trControl = trainControl(method="cv",number=5,index = folds_5)
)
saveRDS(rf_model,"rf_model.rds")
rf_model$resample
res_rf_model <- predict(rf_model,as.matrix(test.data))
postResample(res_rf_model,test.output)
print("======Light GBM===========")
#Defining Categorical Variables for LGBM optimization
categorical_feature <- c("visitNumber","networkDomain","country","source","operatingSystem","deviceCategory","region","browser","metro","city","continent"
                         ,"subContinent","channelGrouping","medium")
train <- lgb.Dataset(data=as.matrix(train.data),label=train.output,categorical_feature =categorical_feature)
print("-------Training LightGBM -------")
lgb_model <- lightgbm(
  data = train
  , learning_rate = .01
  , objective="regression"
  , num_leaves = 48
  , min_data_in_leaf = 30
  , max_depth = 8
  , min_child_samples=20
  , boosting = "gbdt"
  , feature_fraction = 0.8
  , metric = c('rmse','mae')
  , lambda_l1 = 1
  , lambda_l2 = 1
  , verbose = 1
  , seed = 0
  , nrounds = 60000
  , eval_freq = 100,
  , early_stopping_rounds = 200
)
lgb.get.eval.result(lgb_model,"train","RMSE")
lgb.importance(lgb_model, percentage = TRUE) %>% head(10)
res_lgbm <- predict(lgb_model,as.matrix(test.data))
postResample(res_lgbm,test.output)
tree_imp <- lgb.importance(lgb_model, percentage = TRUE)
lgb.plot.importance(tree_imp, top_n = 10, measure = "Gain")
print("---------- Neural Network ----------")
# Normalize training data
# Test data is *not* used when calculating the mean and std.
train.data <- scale(train.data)
# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(train.data, "scaled:center")
col_stddevs_train <- attr(train.data, "scaled:scale")
test.data <- scale(test.data, center = col_means_train, scale = col_stddevs_train)
rmse <- function(y_pred,y_true){
  K <- backend()
  return(K$sqrt(K$mean(K$square(y_pred - y_true))))
}
build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 256, activation = "relu",
                input_shape = dim(train.data)[2]) %>%
    layer_dropout(rate = 0.75) %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dropout(rate = 0.25) %>%
    layer_dense(units = 1)
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list(custom_metric("RMSE",rmse),"accuracy","mean_absolute_error")
  )
  model
}
model <- build_model()
model %>% summary()
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)
history <- model %>% fit(
  as.matrix(train.data),
  train.output,
  epochs = 100,
  batch_size=256,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(callback_early_stopping(patience = 5),print_dot_callback)
)
#NN Test Output
model %>% evaluate(train.data, train.output)
test_predictions <- model %>% predict(test.data)
caret_model_resample <- resamples(list(LM=linear_model,DTREE=dtree_model1,BAGGING=bag_model,RF=rf_model))
bwplot(caret_model_resample, metric="RMSE")