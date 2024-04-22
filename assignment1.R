rm(list=ls())
library(ISLR)
library(class)
library(pROC)
library(randomForest)
set.seed(4061)
### loading the data 3(a)###
x = Smarket[,-9]
y = as.factor(Smarket$Direction)
train = sample(1:nrow(Smarket),1000)
data_rf = data.frame(x, y)
### Splitting data into train and test sets ###
train_data = data_rf[train,]
test_data = data_rf[-train,]
### Fitting Random Forest to the training data ###
rf.out = randomForest(y~., train_data)
rf.yhat.train = predict(rf.out, train_data, type="class")
tb.rf.train = table(rf.yhat.train, train_data$y)
tb.rf.train
### Performing predictions on the test data 3(b)###
rf.yhat.test = predict(rf.out, test_data, type = "class")
tb.rf.test = table(rf.yhat.test, test_data$y)
tb.rf.test
### Plot ROC Curve for the test data ###
rf.test.probs = predict(rf.out, test_data, type="prob")
roc.rf = roc(response=(test_data$y), predictor=rf.test.probs[,1])
roc.rf$auc
plot(roc.rf)
### Preparing data for kNN model 3(c)###
x.train = x[train,]
x.test = x[-train,]
y.train = y[train]
y.test = y[-train]
### fitting the data to kNN ###  
K = 2
ko = knn(x.train, x.test, y.train, K)
knn.pred = as.numeric(ko == "Up")
tb.knn = table(ko, y.test)
tb.knn
### ROC curve for kNN ###
knn.probs = attributes(knn(x.train, x.test, y.train, K, prob=TRUE))$prob
new.knn.probs = 1 - knn.probs
knn.probs.final = ifelse(knn.pred == 1, knn.probs, new.knn.probs)
roc.knn = roc(y.test ,knn.probs.final)
roc.knn$auc
plot(roc.knn, add = TRUE, col = "orange")
legend("bottomright",legend = c("Random Forest", "KNN"), col = c("black", "orange"), lwd = 5)

### KNN classification for different k values 3(d)###
set.seed(4061)
M = 1000
train = sample(1:nrow(Smarket), M)
x.train = x[train,]
x.test = x[-train,]
y.train = y[train]
y.test = y[-train]
Kmax = 10
acc = numeric(Kmax)
for(k in 1:Kmax){
  ko = knn(x.train, x.test, y.train, k)
  tb = table(ko, y.test)
  acc[k] = sum(diag(tb)) / sum(tb)	
}
misclass.err = 1 - acc
plot(misclass.err, pch=20, t='b', 
     main = "Plot of K vs Misclassification Error Rate",
     ylab = "Misclassification Error Rate", xlab="K")
