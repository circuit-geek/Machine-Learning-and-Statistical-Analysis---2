############# Question - 1 #############
rm(list=ls())
library(ISLR)
library(gbm)
df = na.omit(Hitters)
df$Salary = log(df$Salary)
rates = c(0.001, 0.05, 0.01, 0.1)
len_rates = length(rates)
set.seed(4061)

B = 100
rmse_val = matrix(NA,nrow=B,ncol=len_rates)
n = nrow(df)

for(i in 1:len_rates) {
  for(j in 1:B){
    idx = sample(c(1:n), size = n, replace = TRUE)
    bdf = df[idx,]
    gb.out = gbm(Salary~., data=bdf, 
                 distribution="gaussian",
                 shrinkage = rates[i])
    gb.pred = predict(gb.out, df[-idx,])
    y_true = df[-idx,]$Salary
    rmse_val[j,i] = sqrt(mean((gb.pred-y_true)^2))
  }
}

rmse_val
mean_rmse_val = numeric(length = len_rates)

for(k in (1:len_rates)){
  mean_rmse_val[k] = mean(rmse_val[,k])
}
mean_rmse_val
colnames(rmse_val) = rates
boxplot(rmse_val, col="orange", main="Boxplot of Shrinkage Rates vs OOB RMSE Values", 
        xlab="Shrinkage Rates", ylab="OOB RMSE")

######### Question - 2 ##########

rm(list = ls())
library(neuralnet)
library(caret)
library(DataExplorer)
setwd("D:/UCC/Academic Work/Sem-2/ST6041 - ML 2/Assignment/Assignment-2")
df = read.csv(file="uws.csv", stringsAsFactors=TRUE)
subdf = df[,c("grade","sex","age","x.mean","x.max")]
y = df$grade
x = df
x$grade = NULL

plot_bar(df)
plot_boxplot(subdf, by="grade")
plot_boxplot(subdf, by="sex")

x_age = subdf$age
y_sex = as.numeric(subdf$sex)
wilcox.test(x_age ~ y_sex, alternative = "two.sided")
x_max = subdf$x.max
wilcox.test(x_max ~ y_sex, alternative = "two.sided")
x_mean = subdf$x.mean
wilcox.test(x_mean ~ y_sex, alternative = "two.sided")


### Means before scaling ###
mean(df$age)
mean(df$x.max)
mean(df$x.mean)

### Means after scaling ###

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

datss = data.frame(lapply(df,myrecode))
datss = data.frame(lapply(datss,myscale))

mean(datss$age)
mean(datss$x.max)
mean(datss$x.mean)

datss$grade = NULL
### Fitting Neural Network Model ###

set.seed(4061)
nno = neuralnet(y~., data = datss, hidden=c(5), linear.output = FALSE)
plot(nno)
nno$result.matrix["error",]
preds = predict(nno, datss, type='class')
final_preds = max.col(preds)
tbp = table(final_preds, y)
sum(diag(tbp))/sum(tbp)
y_confusion = as.numeric(y)
confusionMatrix(as.factor(final_preds), as.factor(y_confusion))

x$sex = as.numeric(x$sex)
x_cor = cor(x)
thresh = 0.95
unique_vals = c()
for (i in colnames(x_cor)) {
  for (j in colnames(x_cor)){
    if ((i!= j) & (abs(x_cor[i,j]) > thresh)) {
      unique_vals = c(unique_vals, c(i,j))
    }
  } 
}
print(unique(unique_vals))
