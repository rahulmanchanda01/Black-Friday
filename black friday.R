#Practice using Analytics Vidya blog
path <- "C:/Users/manch/Desktop/R class/train_oSwQCTC"
setwd(path)
install.packages("data.table")
library(data.table)
train <- data.table::fread("train.csv", stringsAsFactors = T)
test <- data.table::fread("test.csv", stringsAsFactors = T)
dim(train)
dim(test)
class(train)
str(train)

summary(train)
summary(test)
test[,Purchase := mean(train$Purchase)]
c <- list(train, test)
combin <- rbindlist(c)

#Gender
combin[,prop.table(table(Gender))] 

#Age
combin[,prop.table(table(Age))]

#City
combin[,prop.table(table(City_Category))]

#Stay in Current Years Variable

combin[,prop.table(table(Stay_In_Current_City_Years))]

#unique in id
length(unique(combin$Product_ID))
length(unique(combin$User_ID))

#missing values

colSums(is.na(combin))

#Data manipulation
combin[,Product_Category_2_NA := ifelse(sapply(combin$Product_Category_2, is.na) == TRUE,1,0)]
combin[,Product_Category_3_NA := ifelse(sapply(combin$Product_Category_3, is.na) ==  TRUE,1,0)]


#Imputation
combin[,Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "-999",  Product_Category_2)]
combin[,Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "-999",  Product_Category_3)]

levels(combin$Stay_In_Current_City_Years)[levels(combin$Stay_In_Current_City_Years) ==  "4+"] <- "4"

levels(combin$Age)[levels(combin$Age) == "0-17"] <- 0
levels(combin$Age)[levels(combin$Age) == "18-25"] <- 1
levels(combin$Age)[levels(combin$Age) == "26-35"] <- 2
levels(combin$Age)[levels(combin$Age) == "36-45"] <- 3
levels(combin$Age)[levels(combin$Age) == "46-50"] <- 4
levels(combin$Age)[levels(combin$Age) == "51-55"] <- 5
levels(combin$Age)[levels(combin$Age) == "55+"] <- 6

combin$Age <- as.numeric(combin$Age)

combin[, Gender := as.numeric(as.factor(Gender)) - 1]
combin[, User_Count := .N, by = User_ID]
combin[, Product_Count := .N, by = Product_ID]


combin[, Mean_Purchase_Product := mean(Purchase), by = Product_ID]
combin[, Mean_Purchase_User := mean(Purchase), by = User_ID]

library(dummies)
combin <- dummy.data.frame(combin, names = c("City_Category"), sep = "_")

str(combin)

combin$Product_Category_2 <- as.integer(combin$Product_Category_2)
combin$Product_Category_3 <- as.integer(combin$Product_Category_3)


#Model building in H2O

c.train <- combin[1:nrow(train),]
c.test <- combin[-(1:nrow(train)),]

c.train <- c.train[c.train$Product_Category_1 <= 18,]

library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()

train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)
colnames(train.h2o)

y.dep <- 14
x.indep <- c(3:13,15:20)
regression.model <- h2o.glm( y = y.dep, x = x.indep, training_frame = train.h2o, family = "gaussian",)

h2o.performance(regression.model)
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase =  predict.reg$predict)
write.csv(sub_reg, file = "sub_reg.csv", row.names = F)



#Random Forest
system.time(
  rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122)
)

system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o)))

#writing submission file
sub_rf <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase =  predict.rforest$predict)
write.csv(sub_rf, file = "sub_rf.csv", row.names = F)



#GBM

system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)
)
h2o.performance (gbm.model)
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm.csv", row.names = F)


#Deep learning

# activation_opt <- c("Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
# l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
# l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
# hyper_params <- list(activation = activation_opt,
#                      l1 = l1_opt,
#                      l2 = l2_opt)
# search_criteria <- list(strategy = "RandomDiscrete", 
#                         max_runtime_secs = 120)
# 
# dl_grid <- h2o.grid("deeplearning", x = x.indep, y = y.dep,
#                     grid_id = "dl_grid",
#                     training_frame = train.h2o,
#                     validation_frame = train.h2o,
#                     seed = 1,
#                     hidden = c(10,10),
#                     hyper_params = hyper_params,
#                     search_criteria = search_criteria)
# 
# dl_gridperf <- h2o.getGrid(grid_id = "dl_grid")
# print(dl_gridperf)


# with optimum parameters

system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.indep,
                                      training_frame = train.h2o,
                                      epoch = 60,
                                      hidden = c(100,100),
                                      activation = "Rectifier",
                                      seed = 1122
  )
)

predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))

#create a data frame and writing submission file
sub_dlearning <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.dl2$predict)
write.csv(sub_dlearning, file = "sub_dlearning_new.csv", row.names = F)




