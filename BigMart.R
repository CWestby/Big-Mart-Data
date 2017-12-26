library(dplyr)
library(ggplot2)
library(caret)
library(VIM)
library(gridExtra)

#Loading data
big_mart <- read.csv("train-file.csv")

#Previewing data
glimpse(big_mart)

head(big_mart)

summary(big_mart)

#Transforming "low fat" and "LF" to "Low Fat"
index <- which(big_mart$Item_Fat_Content == "LF" | 
                 big_mart$Item_Fat_Content == "low fat")

big_mart[index, "Item_Fat_Content"] <- "Low Fat"

#Transforming "reg" to "Regular
index2 <- which(big_mart$Item_Fat_Content == "reg")

big_mart[index2, "Item_Fat_Content"] <- "Regular"

#Dropping Unused Levels
big_mart$Item_Fat_Content <- droplevels(big_mart$Item_Fat_Content)
big_mart$Item_Fat_Content <- factor(big_mart$Item_Fat_Content)

#Using kNN imputation for missing values
big_mart_imputed <- kNN(big_mart)
big_mart_imputed <- big_mart_imputed %>% 
    select(Item_Identifier:Item_Outlet_Sales)

summary(big_mart_imputed)

#Finding a Way To Fill Missing Info in Outlet_Size Column
big_mart_q <- big_mart_imputed %>%
  filter(Outlet_Size == "")

table(big_mart_q$Outlet_Type)

big_mart_high <- big_mart_imputed %>%
  filter(Outlet_Size == "High")

table(big_mart_high$Outlet_Type)

big_mart_medium <- big_mart_imputed %>%
  filter(Outlet_Size == "Medium")

table(big_mart_medium$Outlet_Type)

big_mart_small <- big_mart_imputed %>%
  filter(Outlet_Size == "Small")

table(big_mart_small$Outlet_Type)

index3 <- which(big_mart_imputed$Outlet_Size == "" & 
                  big_mart_imputed$Outlet_Type == 
                  "Grocery Store")
big_mart_imputed[index3, "Outlet_Size"] <- "Small"

table(big_mart_imputed$Outlet_Size, 
      big_mart_imputed$Outlet_Identifier)

table(big_mart_imputed$Outlet_Size, 
      big_mart_imputed$Outlet_Type)

#All Outlets marked by the same Outlet_Identifier are the same Outlet_Size
#Outlet Types of Supermarket Type 1 of all sizes. Sizes that are high are a small amount
#Changing one outlet to size High and one to size Medium
index4 <- which(big_mart_imputed$Outlet_Identifier == "OUT017")
big_mart_imputed[index4, "Outlet_Size"] <- "Small"

index5 <- which(big_mart_imputed$Outlet_Identifier == "OUT045")
big_mart_imputed[index5, "Outlet_Size"] <- "Medium"

#Dropping Unused Levels from Outlet_Identifier Column
big_mart_imputed$Outlet_Size <- factor(big_mart_imputed$Outlet_Size)

summary(big_mart_imputed)

#Graphing Data
#Sales Histogram by Outlet Size
ggplot(big_mart_imputed, aes(x=Item_Outlet_Sales, 
                             fill = Outlet_Size)) +
  geom_histogram(binwidth = 200) +
  facet_grid(. ~ Outlet_Size) +
  labs(title = "Item Outlet Sales Histogram", 
       x = "Item Outlet Sales")

#Sales Histogram by Outlet Type
ggplot(big_mart_imputed, aes(x=Item_Outlet_Sales, 
                             fill = Outlet_Type)) +
  geom_histogram(binwidth = 200) +
  facet_wrap(~ Outlet_Type) +
  labs(title = "Item Outlet Sales Histogram", 
       x = "Item Outlet Sales")

#Boxplot Item Outlet Sales by Outlet Size
ggplot(big_mart_imputed, aes(x = Outlet_Identifier,
                             y = Item_Outlet_Sales)) +
  geom_boxplot() +
  labs(title = "Sales by Outlet Size",
       x = "Outlet Identifier",
       y = "Item Outlet Sales")

#Plot of Item Outlet Sales by Item MRP
ggplot(big_mart_imputed, aes(x = Item_MRP,
                             y = Item_Outlet_Sales)) +
  geom_bin2d() +
  facet_grid(. ~ Outlet_Size) +
  labs(title = "Item Outlet Sales by Item MRP and Outlet Size",
       x = "Item MRP",
       y = "Item Outlet Sales")

#Plot of Item Outlet Sales by Item MRP and Outlet Type
ggplot(big_mart_imputed, aes(x = Item_MRP,
                             y = Item_Outlet_Sales)) +
  geom_bin2d() +
  facet_wrap(~ Outlet_Identifier) +
  labs(title = "Item Outlet Sales by Item MRP and Outlet Type",
       x = "Item MRP",
       y = "Item Outlet Sales")

#Further Investigation 
#Median Sales by Outlet Type
big_mart_imputed %>%
  group_by(Outlet_Type) %>%
  summarize(median(Item_Outlet_Sales))

#Median Sales by Outlet Size
big_mart_imputed %>%
  group_by(Outlet_Size) %>%
  summarize(median(Item_Outlet_Sales))

#Median Sales by Outlet Size and Outlet Type
big_mart_imputed %>%
  group_by(Outlet_Size, Outlet_Type) %>%
  summarize(median(Item_Outlet_Sales))

#Median Sales by Outlet Identifier
big_mart_imputed %>%
  group_by(Outlet_Identifier, Item_Identifier) %>%
  summarize(median(Item_Outlet_Sales))

big_mart_imputed %>%
  group_by(Outlet_Identifier, Outlet_Size) %>%
  summarize(median(Item_Outlet_Sales))

big_mart_imputed %>%
  group_by(Outlet_Identifier, Outlet_Type) %>%
  summarize(median(Item_Outlet_Sales))

big_mart_grouped <- big_mart_imputed %>%
  group_by(Outlet_Identifier, Item_Identifier) %>%
  summarize(median(Item_Outlet_Sales))

#Preparing Data For Machine Learning
big_mart_sub <- big_mart_imputed %>%
  select(-Item_Identifier, -Outlet_Identifier)


#Machine Learning Model
#Partitioning Data
inTrain <- createDataPartition(y = big_mart_sub$Item_Outlet_Sales, 
                               p = 0.7, list=FALSE)
train <- big_mart_sub[inTrain, ]
test <- big_mart_sub[-inTrain, ]

#LM Model
model_lm <- train(Item_Outlet_Sales ~ ., train, 
                  method = "lm", 
                  preProcess = c("center", "scale"), 
                  trControl = trainControl(method = "cv", 
                                           number = 10))
model_lm


predictions_lm <- predict(model_lm, test)
error <- predictions_lm - test$Item_Outlet_Sales

# Calculate RMSE
sqrt(mean((predictions_lm - 
             big_mart_sub$Item_Outlet_Sales)^2))

#GLMNET Model
model_glmnet <- train(Item_Outlet_Sales ~ ., train, 
                      method = "glmnet", 
                      preProcess = c("center", "scale"), 
                      trControl = trainControl(method = "cv", 
                                               number = 10))
model_glmnet

predictions_glmnet <- predict(model_glmnet, test)
error <- predictions_glmnet - test$Item_Outlet_Sales

# Calculate RMSE
sqrt(mean((predictions_glmnet - 
             big_mart_sub$Item_Outlet_Sales)^2))

#GLM Model
model_glm <- train(Item_Outlet_Sales ~ ., train, 
                  method = "glm", 
                  preProcess = c("center", "scale"), 
                  trControl = trainControl(method = "cv", 
                                           number = 10))
model_glm

predictions_glm <- predict(model_glm, test)
error <- predictions_glm - test$Item_Outlet_Sales

# Calculate RMSE
sqrt(mean((predictions_glm - 
             big_mart_sub$Item_Outlet_Sales)^2))


#Testing Model
testing <- read.csv("test-file.csv")

#Transforming "low fat" and "LF" to "Low Fat"
index <- which(testing$Item_Fat_Content == "LF" | 
                 testing$Item_Fat_Content == "low fat")

testing[index, "Item_Fat_Content"] <- "Low Fat"

#Transforming "reg" to "Regular
index2 <- which(testing$Item_Fat_Content == "reg")

testing[index2, "Item_Fat_Content"] <- "Regular"

#Dropping Unused Levels
testing$Item_Fat_Content <- droplevels(testing$Item_Fat_Content)
testing$Item_Fat_Content <- factor(testing$Item_Fat_Content)

#Using kNN imputation for missing values
testing_imputed <- kNN(testing)
testing_imputed <- testing_imputed %>% 
  select(Item_Identifier:Outlet_Type)

summary(testing_imputed)

index3 <- which(testing_imputed$Outlet_Size == "" & 
                  testing_imputed$Outlet_Type == 
                  "Grocery Store")
testing[index3, "Outlet_Size"] <- "Small"

index4 <- which(testing_imputed$Outlet_Identifier == "OUT017")
testing_imputed[index4, "Outlet_Size"] <- "Small"

index5 <- which(testing_imputed$Outlet_Identifier == "OUT045")
testing_imputed[index5, "Outlet_Size"] <- "Medium"

#Dropping Unused Levels from Outlet_Identifier Column
testing_imputed$Outlet_Size <- factor(testing_imputed$Outlet_Size)

testing_predictions_lm <- predict(model_lm, testing_imputed)

testing_imputed$Item_Outlet_Sales <- testing_predictions_lm

submission_lm <- testing_imputed[, c("Item_Identifier",
                                      "Outlet_Identifier",
                                      "Item_Outlet_Sales")]
write.csv(submission_lm, "big_mart_predictions.csv", 
          row.names = FALSE)
