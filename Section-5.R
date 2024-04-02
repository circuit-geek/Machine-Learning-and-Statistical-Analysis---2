################ Section - 5 #####################

####### Exercise - 1 #######
rm(list=ls())
library(ISLR)
library(e1071)
library(pROC)
library(caret)
df = Carseats
### Converting into binary classification problem
High = ifelse((df$Sales<=8),'No', 'Yes')
### Combining the dataframe with the labels
CS = data.frame(df, High)
### Separating X and y having the data and labels 
CS$Sales = NULL
X = CS
y = CS$High
X$High = NULL
### Spliting data into train and test splits
set.seed(4061)
n = nrow(X)
idx = sample(1:n , 350)
X.train = X[idx, ]
X.test = X[-idx, ]
y.train = y[idx]
y.test = y[-idx]
### Fitting SVM model (from e1071 package)
svmo = svm(X.train, y.train, kernel='polynomial')
### Error occuring here is because the covariates cannot be supported
### in factor category, so perform one hot encoding to convert them.
new_data = model.matrix(y~.+0, data = X)
new.X.train = new_data[idx, ]
new.X.test = new_data[-idx, ]
y.train = as.factor(y.train)
### Now fitting the SVM model 
svm.p = svm(new.X.train, y.train, kernel = 'polynomial')
summary(svm.p)
svm.l = svm(new.X.train, y.train, kernel = 'linear')
summary(svm.l)

### Evaluating model on train data 
svm.p.train_preds = fitted(svm.p)
svm.l.train_preds = fitted(svm.l)
table(y.train, svm.p.train_preds)
table(y.train, svm.l.train_preds)

### Evaluating model on test data 
svm.p.test_preds = predict(svm.p, newdata = new.X.test)
svm.l.test_preds = predict(svm.l, newdata = new.X.test)
y.test = as.factor(y.test)
table(y.test, svm.p.test_preds)
table(y.test, svm.l.test_preds)

### In order to generate probabilities we have to specify probability = TRUE
### while fitting/training the model
svm.p.probs = svm(new.X.train, y.train, kernel = 'polynomial', probability = TRUE)
svm.l.probs = svm(new.X.train, y.train, kernel = 'linear', probability = TRUE)

### Now testing on test data
svm.p.probs_test = predict(svm.p.probs, newdata = new.X.test, probability = TRUE)
svm.l.probs_test = predict(svm.l.probs, newdata = new.X.test, probability = TRUE)

### Computing the confusion matrix using the caret package 
confusionMatrix(data = svm.p.probs_test, reference = y.test, positive = 'Yes')
confusionMatrix(data = svm.l.probs_test, reference = y.test, positive = 'Yes')

### Computing the AUC value (we need to extract P(Y=1|X))
prob.lin = attributes(svm.l.probs_test)$probabilities[,2]
prob.pol = attributes(svm.p.probs_test)$probabilities[,2]
y.test = as.factor(y.test)
roc(response=y.test, predictor=prob.lin)$auc
roc(response=y.test, predictor=prob.pol)$auc
plot(roc(response=y.test, predictor=prob.lin))

####### Exercise - 2 #######

rm(list=ls())
set.seed(4061)
data = iris
n = nrow(data)
idx = sample(1:n, 100)
X = data
y = data$Species
X$Species = NULL

X.train = X[idx, ]
X.test = X[-idx, ]
y.train = y[idx]
y.test = y[-idx]

svm.p = svm(X.train, y.train, kernel = 'polynomial')
svm.l = svm(X.train, y.train, kernel = 'linear')
svm.r = svm(X.train, y.train, kernel = 'radial')

summary(svm.p) ## 44 support vectors
summary(svm.l) ## 26 support vectors
summary(svm.r) ## 44 support vectors

svm.p.preds = predict(svm.p, newdata = X.test)
svm.l.preds = predict(svm.l, newdata = X.test)
svm.r.preds = predict(svm.r, newdata = X.test)

table(svm.p.preds, y.test)
table(svm.l.preds, y.test)
table(svm.r.preds, y.test)

confusionMatrix(svm.p.preds, y.test) ## 94% accuracy
confusionMatrix(svm.l.preds, y.test) ## 98% accuracy
confusionMatrix(svm.r.preds, y.test) ## 100% accuracy (Before tuning)

set.seed(4061)
svm.tune = e1071::tune(svm, train.x=X.train, train.y=y.train,
                       kernel='radial',
                       ranges=list(cost=10^(-2:2), 
                       gamma=c(0.5,1,1.5,2)))
print(svm.tune)
names(svm.tune)

bp = svm.tune$best.parameters

new.svm = svm(X.train, y.train, kernel = 'radial', cost = bp$cost, gamma =  bp$gamma)
new.svm.preds = predict(new.svm, newdata = X.test)
confusionMatrix(new.svm.preds, y.test) ## 96% accuracy (After tuning)

####### Exercise - 3 #######

rm(list=ls())
set.seed(4061)
data = Hitters
data = na.omit(data)
n = nrow(data)
data$Salary = as.factor(ifelse(data$Salary>median(data$Salary),
                              "High","Low"))

X = data
y = data$Salary
X$Salary = NULL

idx = sample(1:n, size = 0.7*n)
X.train = X[idx, ]
X.test = X[-idx, ]
y.train = y[idx]
y.test = y[-idx]

## Since the data has many categorical data, performing one hot 
## encoding
new_data = model.matrix(y~.+0, data = X)
new.X.train = new_data[idx, ]
new.X.test = new_data[-idx, ]
y.train = as.factor(y.train)
y.test = as.factor(y.test)

svm.l = svm(new.X.train, y.train, kernel = 'linear')
svm.r = svm(new.X.train, y.train, kernel = 'radial')
svm.l.preds = predict(svm.l, newdata = new.X.test)
svm.r.preds = predict(svm.r, newdata = new.X.test)
confusionMatrix(svm.l.preds, y.test) ## 79.75% accuracy - Linear
confusionMatrix(svm.r.preds, y.test) ## 82.28% accuracy - Radial

####### Exercise - 4 (SVM Regression Problem) #######

rm(list=ls())
set.seed(4061)

data = iris
data$Species = NULL
n = nrow(data)
X = data
y = data$Sepal.Length
X$Sepal.Length = NULL

idx = sample(1:n, 100)

X.train = X[idx, ]
X.test = X[-idx, ]
y.train = y[idx]
y.test = y[-idx]

train.data = data.frame(X.train, y.train)
train_control = trainControl(method = "cv", number = 10)
model = train(y.train~., data = train.data, method = "svmRadial", trControl = train_control)
model

svm.p = predict(model, newdata=X.test)
mean( (y.test-svm.p)^2 )