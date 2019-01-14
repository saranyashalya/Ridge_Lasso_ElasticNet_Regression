library(glmnet)
library(caret)
library(mlbench)
library(psych)

data("BostonHousing")
data <- BostonHousing

str(data)

pairs.panels(data[,-c(14,4)])

#Data partition
library(caTools)
set.seed(222)
sample =sample.split(data$medv, SplitRatio = 0.7)
train <- data[sample==TRUE,]
test <- data[sample==FALSE,]

#custom control parameters
custom = trainControl(method = "repeatedcv", 
                      number = 10,
                      repeats = 5,
                      verboseIter = T)

#Linear model
set.seed(1234)

linear_model <- train(medv~., 
      data = train, 
      method="lm",
      trControl = custom)

linear_model$results
linear_model
summary(linear_model)
plot(linear_model$finalModel)


#Ridge Regression
set.seed(1234)

ridge_model <- train(medv ~ .,
                     train,
                     method="glmnet",
                     tuneGrid = expand.grid(alpha=0,
                                            lambda = seq(0.0001, 1 , length = 5)),
                      trControl=custom)

plot(ridge_model)

ridge_model
plot(ridge_model$finalModel, xvar = "lambda", label= T)
plot(ridge_model$finalModel, xvar="dev", label = T)
plot(varImp(ridge_model, scale = F))


##Lasso Regression
set.seed(1234)

lasso_model <- train(medv ~ .,
                     train,
                     method="glmnet",
                     tuneGrid = expand.grid(alpha=1,
                                            lambda = seq(0.0001, 0.2 , length = 5)),
                     trControl=custom)

# plot results
plot(lasso_model)
lasso_model
plot(lasso_model$finalModel, xvar="lambda", label = T)
plot(lasso_model$finalModel, xvar="dev", label=T)
plot(varImp(lasso_model, scale=F))

##Elastic net regression
set.seed(1234)
en_model <- train(medv~.,
                  train,
                  method="glmnet",
                  tuneGrid = expand.grid(alpha=seq(0,1, length=10),
                                        lambda = seq(0.0001, 1, length=5)),
                  trControl = custom)

#plot results
plot(en_model)
plot(en_model$finalModel, xvar ="lambda", label=T)
plot(en_model$finalModel, xvar ="dev", label=T)

# compare models
model_list <- list(LinearModel = linear_model, ridge = ridge_model, lasso =lasso_model, 
                   elastic = en_model)

res <- resamples(model_list)
summary(res)

bwplot(res)
xyplot(res, metric="RMSE")

# BEst model
en_model$bestTune
 best <- en_model$finalModel
coef(best, s= en_model$bestTune$lambda) 

# Save model for later use
saveRDS(en_model, "C:\\Users\\H303937\\OneDrive - Honeywell\\my_code\\final_model.rds")
fm <- readRDS("final_model.rds")
print(fm)

## Prediction
p1 <- predict(fm, train)
sqrt(mean((train$medv-p1)^2))

p2 <- predict(fm, test)
sqrt(mean((test$medv-p2)^2))
