library(ISLR)
setwd("E:/北大新资料/统计学习")
Spam.train <- read.table(file.path(getwd(), "spam.train"))
n_1=0
for(i in 1:4601)
{
  if(Spam.train$V58[i]==1)
    n_1=n_1+1
}
summary(Spam.train)
glm.fit=glm(V58~.,data=Spam.train,family=binomial)
glm.probs=predict(glm.fit,newdata=Spam.train)
glm.pred=rep(0,4601)
glm.pred[glm.probs>.5]=1
err=glm.pred-Spam.train$V58
n01=0
n10=0
for(i in 1:4601)
{
  if(err[i]==-1)
    n01=n01+1
  if(err[i]==1)
    n10=n10+1
}
#随机森林
library(randomForest)
rd.fit=randomForest(as.factor(V58)~.,data=Spam.train)
rd.fit

