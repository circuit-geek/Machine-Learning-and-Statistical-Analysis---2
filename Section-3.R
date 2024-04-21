############### Section-3 ##################

########## Exercise - 1 (KNN) ###########
rm(list=ls())
set.seed(4061)
library(class)
library(caret)

df = iris
n = nrow(df)
idx = sample(1:n, )
shuffled_df = df[idx,]
X = shuffled_df
y = shuffled_df$Species
X$Species = NULL
itrain = sample(1:nrow(shuffled_df), 0.8*nrow(shuffled_df))
X.train = X[itrain,]
X.test = X[-itrain,]
y.train = y[itrain]
y.test = y[-itrain]

K = 1
knn_model = knn(X.train, X.test, y.train, K)
knn_model ### Returns the predicted classes for X.test!!

tb = table(knn_model, y.test)
tb
confusionMatrix(data = knn_model, reference = y.test)

Kmax = 30
acc = numeric(length = Kmax)

for(i in 1:Kmax) {
  knn_new_model = knn(X.train, X.test, y.train, i)
  tb = table(knn_new_model, y.test)
  acc[i] = sum(diag(tb))/sum(tb)
}
plot(1-acc, pch=20, t='b', xlab='k')

########### Exercise-2 (Logistic Regression) #########
rm(list=ls())
set.seed(4061)
df = iris
n = nrow(df)
shuffled_df = df[sample(1:n),] 
x = shuffled_df
x$is.virginica = as.numeric(x$Species=="virginica")
x$Species = NULL
idx = sample(1:nrow(x), 100)
X.train = x[idx,]
X.test = x[-idx,]
y.test = X.test$is.virginica
X.test$is.virginica = NULL

log_model = glm(is.virginica~.,data = X.train, family = "binomial")
summary(log_model)

log_preds = predict(log_model, newdata = X.test, type = "response")
boxplot(log_preds~y.test)
preds = as.factor(as.numeric(log_preds>.5))
confusionMatrix(data=preds, reference=as.factor(y.test), positive="1")

############### Exercise - 3 (LDA-Assumptions) ###############
library(MASS) ## for LDA
rm(list=ls())
set.seed(4061)
df = iris
attach(df)
df$is.virginica = as.numeric(df$Species=="virginica")
df$Species = NULL

### Assumptions of LDA ###
# 1) Predictor variables for all the classes are normally distributed
# 2) Different classes have the same covariance matrix.(Homoscedacity)
##########################
## We can use Bartlett's test to check equal covariance ##
## We can use Shapiro's test to check for Normal Distribution of data ##

for(i in 1:4){
  print(bartlett.test(df[,i]~df$is.virginica)$p.value)
}

## H0: All the variance are equal (1st Feature => 0.94)
## Ha: Variances are not equal (2=>0.0025, 3=>1.94*10^-11, 4=>1.695*10^-7)

for(i in 1:4){
  boxplot(df[,i]~df$is.virginica)
}

for(i in 1:4){
  print(shapiro.test(df[which(df$is.virginica==1),i])$p.value)
} ## Normally Distributed
 
for(j in 1:4){
  print(shapiro.test(df[which(df$is.virginica==0),j])$p.value)
} ## Only 1 variable normally distributed.
 
lda.model = lda(is.virginica~., data = df)
lda.model ## (DS: -0.19, 0.87, 0.017, 2.39)

## Interpretation for the model fit is that based on the discriminant
## score for the coefficients, Petal.Width plays the major role 
## in creating a linear boundary between the two classes, followed
## by Sepal.Width which takes the second most precendence.

#################### Exercise - 4 (LDA) ##################

rm(list=ls())
library(MASS) ## for LDA
set.seed(4061)
df = iris
attach(df)
df$is.virginica = as.numeric(df$Species=="virginica")
df$Species = NULL
X = df
X$is.virginica = NULL
y = df$is.virginica
idx = sample(1:nrow(df), 100)
X.train = X[idx,]
X.test = X[-idx,]
y.train = y[idx]
y.test = y[-idx]

lda_model = lda(y.train ~., data = X.train)
lda_model
###
#Positive Coefficient:
#A positive coefficient means that as the value of that predictor variable increases, the corresponding linear discriminant value also increases.
#In terms of classification, this means that higher values of the predictor variable tend to push observations towards that class.
#Negative Coefficient:
#A negative coefficient means that as the value of that predictor variable increases, the corresponding linear discriminant value decreases.
#In terms of classification, this means that higher values of the predictor variable tend to push observations away from that class.
###

lda_preds = predict(lda_model, newdata = X.test)
lda_preds$class

tb = table(lda_preds$class, y.test)
acc_lda = sum(diag(tb))/sum(tb)
acc_lda ## 98%

qda_model = qda(y.train~., data=X.train)
qda_preds = predict(qda_model , newdata=X.test)

tb = table(qda_preds$class, y.test)
acc_qda = sum(diag(tb))/sum(tb)
acc_qda ## 98%

########## Differences between LDA and QDA #############
#Covariance Matrix:
#LDA: Assumes all classes share the same covariance matrix.
#QDA: Allows each class to have its own covariance matrix.
#Decision Boundary:
#LDA: Linear decision boundary (straight line or plane).
#QDA: Quadratic decision boundary (curved surface).
#Flexibility:
#LDA: Less flexible, assumes linear relationships.
#QDA: More flexible, can capture non-linear relationships.
#Computational Efficiency:
#LDA: Generally more computationally efficient due to simpler assumptions.
#QDA: Can be more computationally expensive, especially with large datasets and many features.