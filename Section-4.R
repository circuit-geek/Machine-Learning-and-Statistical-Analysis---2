rm(list=ls())

############ Exercise-1 ##############
set.seed(4061)
library(ISLR)
df = Carseats
library(tree)
library(caret)

High = ifelse(df$Sales <= 8, 'No', 'Yes')
CS = data.frame(df, High)
CS$Sales = NULL
CS$High = as.factor(CS$High)

tree.model = tree(High~., CS)
summary(tree.model)
plot(tree.model)
text(tree.model, pretty = 0)

### Tree Pruning ###
set.seed(3)
tree_prune = cv.tree(tree.model, FUN = prune.misclass)
par(mfrow=c(1,2))
plot(tree_prune$size, tree_prune$dev, t='b')
abline(v=tree_prune$size[which.min(tree_prune$dev)])
plot(tree_prune$k, tree_prune$dev, t='b')

opt.size = tree_prune$size[which.min(tree_prune$dev)]
ptree = prune.misclass(tree.model, best=opt.size)
par(mfrow=c(1,1))
summary(ptree)
plot(ptree)
text(ptree, pretty = 0)

############## Exercise-2 ###############

set.seed(4061)
n = nrow(CS)
X = CS
y = CS$High
X$High = NULL
idx = sample(1:n, 200)
X.train = X[idx,]
X.test = X[-idx,]
y.train = y[idx]
y.test = y[-idx]

tree.model_1 = tree(y.train~., X.train)
plot(tree.model_1)
text(tree.model_1, pretty = 0)

preds_tree = predict(tree.model_1, X.train, type = "class")
preds_pruned_tree = predict(ptree, X.train, type = "class")
confusionMatrix(preds_tree, y.train)
confusionMatrix(preds_pruned_tree, y.train)

preds_tree_test = predict(tree.model_1, X.test, type = "class")
preds_pruned_tree_test = predict(ptree, X.test, type = "class")
confusionMatrix(preds_tree_test, y.test)
confusionMatrix(preds_pruned_tree_test, y.test)

library(pROC)
tree.probs = predict(tree.model_1, X.test, type = "vector")
pruned_tree.probs = predict(ptree, X.test, type = "vector")

roc_tree = roc(response=(y.test), predictor=tree.probs[,1])
roc_tree_pruned = roc(response=(y.test), predictor=pruned_tree.probs[,1])
plot(roc_tree)
plot(roc_tree_pruned, add=TRUE, col="green")
roc_tree$auc
roc_tree_pruned$auc

############ Exercise - 3 #############

set.seed(4061)
df = Hitters
df = na.omit(df)
df$Salary = log(df$Salary)
X = df
y = df$Salary
X$Salary = NULL

myrecode <- function(x){
  
  if(is.factor(x)){
    levels(x)
    return(as.numeric(x)) 
  } else {
    return(x)
  }
}

myscale <- function(x){
  minx = min(x,na.rm=TRUE)
  maxx = max(x,na.rm=TRUE)
  return((x-minx)/(maxx-minx))
}

datss = data.frame(lapply(X,myrecode))
datss = data.frame(lapply(datss,myscale))


new_tree = tree(y ~., datss)
plot(new_tree)
text(new_tree, pretty = 0)
summary(new_tree)

############# Exercise - 4 #############

library(randomForest)
set.seed(4061)
df = Carseats
high = ifelse(df$Sales <= 8, 'No', 'Yes')
new_df = data.frame(df, high)
X = new_df
y = new_df$high
y = as.factor(y)
X$Sales = NULL
X$high = NULL
p = ncol(X) 

## Decision Tree Model ##
tree_model = tree(y~., X)
tree_preds = predict(tree_model, X, type = "class")
confusionMatrix(tree_preds, y)
## Random Forest Model ##
forest_model = randomForest(y~., X)
forest_preds = predict(forest_model, X, type = "class")
confusionMatrix(forest_preds, y)
## Bagging ##
bag_model = randomForest(y~., X, mtry=p)
bag_preds = predict(bag_model, X, type = "class")
confusionMatrix(bag_preds, y)

set.seed(4061)
n = nrow(new_df)
X = new_df
y = new_df$high
y = as.factor(y)
X$Sales = NULL
X$high = NULL
idx = sample(1:n, 200)
X.train = X[idx,]
X.test = X[-idx,]
y.train = y[idx]
y.test = y[-idx]
p = ncol(X)

new_tree_model = tree(y.train ~., X.train)
new_tree_preds = predict(new_tree_model, X.test, type="class")
confusionMatrix(new_tree_preds, y.test) ## 76%

new_forest_model = randomForest(y.train~., X.train)
new_forest_preds = predict(new_forest_model, X.test, type="class")
confusionMatrix(new_forest_preds, y.test) ## 83.5% 

new_bag_model = randomForest(y.train~., X.train, mtry=p)
new_bag_preds = predict(new_bag_model, X.test, type="class")
confusionMatrix(new_bag_preds, y.test) ## 85%

########### Exercise - 6 ###########

cbind(forest_model$importance, bag_model$importance)
par(mfrow=c(1,2))
varImpPlot(forest_model, pch=15, main="Ensemble method 1")
varImpPlot(bag_model, pch=15, main="Ensemble method 2")
## If the importance value is high, that mean the variable is 
## important in making accurate predictions.

############ Exercise - 7 ###########

library(gbm)

set.seed(4061)
n = nrow(new_df)
X = new_df
y = new_df$high
y = (as.numeric(y =="Yes"))
X$Sales = NULL
X$high = NULL
idx = sample(1:n, 300)
X.train = X[idx,]
X.test = X[-idx,]
y.train = y[idx]
y.test = y[-idx]

gbm_model = gbm(y.train~., data = X.train, 
                distribution = "bernoulli", n.trees = 5000,
                interaction.depth = 1)
gbm_model_preds = predict(gbm_model, newdata = X.test, 
                          n.trees = 5000)

confusionMatrix(gbm_model_preds, y.test)
roc.gb = roc(response=y.test, predictor=gbm_model_preds)
plot(roc.gb)
roc.gb$auc


########## Exercise - 8 ##########

rm(list=ls()) # clear the environment

# Set up the data (take a subset of the Hitters dataset)
data(Hitters)
Hitters = na.omit(Hitters)
dat = Hitters
# hist(dat$Salary)
dat$Salary = log(dat$Salary)
n = nrow(dat)
NC = ncol(dat)

# Data partition
itrain = sample(1:n, size=round(.7*n))
dat.train = dat[itrain,]
dat.validation = dat[-itrain,]
x = dat.train
x$Salary = NULL
y = dat.train$Salary
ytrue = dat.validation$Salary

gb.out = train(Salary~., data=dat.train, method='gbm', distribution='gaussian')
gb.fitted = predict(gb.out) # corresponding fitted values
gb.pred = predict(gb.out, dat.validation)
mean((gb.pred-ytrue)^2)