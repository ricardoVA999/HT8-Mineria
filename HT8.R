library(caret)
library(rpart)
library(e1071)
library(rpart.plot)
library(neuralnet)
library(dummies)
library(nnet)
library(RWeka)
library(neural)
library(corrplot)

#Trabajando con variables significativas tanto categoricas como cuantitativas
houses = read.csv('train.csv')
houses[is.na(houses)]<-0
houses$Id<-NULL

houses$clasification <- ifelse(houses$SalePrice > 290000, "caras", ifelse(houses$SalePrice>170000, "intermedio", "economicas"))
houses$clasification <- as.factor(houses$clasification)
houses<- houses[,c(4,12,17,34,38,46,62,67,80,81)]


porciento <- 70/100
set.seed(1234)

economicas<-houses[houses$clasification=="economicas",]
intermedias<-houses[houses$clasification=="intermedio",]
caras<-houses[houses$clasification=="caras",]

numFilasTrainEcon<-sample(nrow(economicas), porciento*nrow(economicas))
trainEcon<-economicas[numFilasTrainEcon,]

numFilasTrainInter<-sample(nrow(intermedias), porciento*nrow(intermedias))
trainInter<-intermedias[numFilasTrainInter,]

numFilasTrainCaras<-sample(nrow(caras), porciento*nrow(caras))
trainCaras<-caras[numFilasTrainCaras,]

numFilasAll<-c(numFilasTrainCaras, numFilasTrainInter, numFilasTrainEcon)


training<-rbind(trainInter, trainEcon, trainCaras)
test<-houses[setdiff(rownames(houses),rownames(training)),]

table(training$clasification)
table(test$clasification)

training$SalePrice<-NULL
test$SalePrice<-NULL

#Primer modelo de classificacion, realizaco con con variables significativas tanto categoricas como cunatitativas y la funcion de activacion softmax
modelo.nn2 <- nnet(clasification~.,data = training, size=2, rang=0.1, decay=5e-4, maxit=200)

prediccion2 <- as.data.frame(predict(modelo.nn2, newdata = test[,1:8]))
columnaMasAlta<-apply(prediccion2, 1, function(x) colnames(prediccion2)[which.max(x)])
test$prediccion2<-columnaMasAlta

cfm<-confusionMatrix(as.factor(test$prediccion2),test$clasification)
cfm
