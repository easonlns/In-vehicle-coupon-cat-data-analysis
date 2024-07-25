library(ISLR)
library(glmnet)
library(neuralnet)
library(MLmetrics)
library(dplyr)
library(ggplot2)
library(ggcorrplot)
set.seed(123)
data = read.csv("/Users/laungaisang/Downloads/in-vehicle-coupon-recommendation 2.csv",sep=",",header=T)

#Analysis on column "Y"
# Count the numbers of "0" and "1" in column Y
counts <- table(data$Y)

# Calculate column Y "0" and "1" percentage
percentage_cal <- prop.table(counts)*100

Y_result <- data.frame(Value = names(counts),
                     Count = as.numeric(counts),
                     Percentage = as.numeric(percentage_cal))

print(Y_result)

#Count missing data
missing_counts <- data %>%
  summarise_all(~sum(is.na(.) | . == ""))

print(missing_counts)

# Data cleaning
# Remove empty data within columns
data <- data %>%
  mutate(
    Bar = na_if(Bar, ""),
    CoffeeHouse = na_if(CoffeeHouse, ""),
    CarryAway = na_if(CarryAway, ""),
    RestaurantLessThan20 = na_if(RestaurantLessThan20, ""),
    Restaurant20To50 = na_if(Restaurant20To50, "")
  ) %>%
  filter(
    !is.na(Bar) & !is.na(CoffeeHouse) & !is.na(CarryAway) &
      !is.na(RestaurantLessThan20) & !is.na(Restaurant20To50)
  )


# Drop Columns----
data <- data[, -which(names(data) == "car")] # Drop 'Car'
data <- data[, -which(names(data) == "toCoupon_GEQ5min")]


# Define Dummy Variables----
data$coupon <- as.factor(data$coupon)
coupon_dummies <- model.matrix(~ coupon - 1, data = data)  # '- 1' omits the intercept to keep all dummy variables
data <- cbind(data, coupon_dummies)

#Descriptive data
#destination
ggplot(data, aes(x = destination, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("destination") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Destination") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#passanger
ggplot(data, aes(x = passanger, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("passanger") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Passanger") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#weather
ggplot(data, aes(x = weather, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("weather") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Weather") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#time
ggplot(data, aes(x = time, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("time") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Time") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#coupon
ggplot(data, aes(x = coupon, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("coupon") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Coupon") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#expiration
ggplot(data, aes(x = expiration, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("expiration") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Expiration") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#gender
ggplot(data, aes(x = gender, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("gender") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Gender") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#age
ggplot(data, aes(x = age, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("age") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Age") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#maritalStatus
ggplot(data, aes(x = maritalStatus, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("maritalStatus") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Marital Status") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#education
ggplot(data, aes(x = education, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("education") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Education") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#occupation
ggplot(data, aes(x = occupation, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("occupation") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Occupation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#income
ggplot(data, aes(x = income, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("income") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Income") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Bar
ggplot(data, aes(x = Bar, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("Bar") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Bar") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#CoffeeHouse
ggplot(data, aes(x = CoffeeHouse, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("CoffeeHouse") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("CoffeeHouse") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#CarryAway
ggplot(data, aes(x = CarryAway, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("CarryAway") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("CarryAway") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#RestaurantLessThan20
ggplot(data, aes(x = RestaurantLessThan20, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("RestaurantLessThan20") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("RestaurantLessThan20") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Restaurant20To50
ggplot(data, aes(x = Restaurant20To50, fill = as.factor(Y))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  xlab("Restaurant20To50") +
  ylab("Proportion") +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  labs(fill = "Y") +
  ggtitle("Restaurant20To50") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Columns with integer datatype
summary(data[,c(4,11,20:23)])

#Correlation matrix
corr <- round(cor(data[,c(4,11,20:23)]), 1)
ggcorrplot(corr,
           hc.order = TRUE,
           type = "lower",
           lab = TRUE)

# Train Set & Test Set----
data <- na.omit(data)
ind <- sample(2, nrow(data), replace = TRUE,
              prob = c(0.8, 0.2))
train = data[ind==1, ]
test = data[ind == 2, ]

# Logit Full Model----

# Logit Regression
model <- glm(Y~., family = binomial(link = 'logit'), data = train)
summary(model)
L1 = logLik(model)
L1

model <- glm(Y~ 1, family = binomial(link = 'logit'), data = train)
L0 = logLik(model)
L0

-2*L0+2*L1

qchisq(0.95, 95-1)

# Backward/Forward Selection
Full_backward <- step(model, direction ='backward')
Full_forward <- step(model, direction ='forward')

predmo <- predict(Full_backward, newdata = test)
predclass <- ifelse(predmo> 0.5, 1, 0)
predclass

# Interaction Term----
interaction_1 <- lm(Y ~ couponBar + Bar + couponBar * Bar, data = train)
summary(interaction_1) # Interaction terms between ‘Bar Coupon’ and ‘Bar Frequency’ 

interaction_2 <- lm(Y ~ couponCoffee House + CoffeeHouse + couponCoffee House * CoffeeHouse, data = train)
summary(interaction_2) # Interaction terms for ‘Coffee House Coupon’ and ‘Coffee House Frequency’ 

interaction_3 <- lm(Y ~ couponCarry out & Take away + CarryAway + couponCarry out & Take away * CarryAway, data = train)
summary(interaction_3) # Interaction terms for ‘Carry Away Coupon’ and ‘Carry Away Frequency’ 

interaction_4 <- lm(Y ~ couponRestaurant(<20) + RestaurantLessThan20 + couponRestaurant(<20) * RestaurantLessThan20, data = train)
summary(interaction_4) # Interaction terms for ‘Restaurant(<20) Coupon’ and ‘Restaurant(<20) Frequency’ 

interaction_5 <- lm(Y ~ couponRestaurant(20-50) + Restaurant20To50 + couponRestaurant(20-50) * Restaurant20To50, data = train)
summary(interaction_5)

bind_interaction <- lm(Y ~ couponBar + Bar + couponBar * Bar + 
                         couponCoffee House + CoffeeHouse + couponCoffee House * CoffeeHouse +
                         couponCarry out & Take away + CarryAway + couponCarry out & Take away * CarryAway +
                         couponRestaurant(<20) + RestaurantLessThan20 + couponRestaurant(<20) * RestaurantLessThan20 +
                         couponRestaurant(20-50) + Restaurant20To50 + couponRestaurant(20-50) * Restaurant20To50, data = train)
summary(bind_interaction)

# LASSO----
VC = data[complete.cases(data),]
x = model.matrix(Y~., VC)[,-1]
y = VC$Y

cv.out = cv.glmnet(x,y,alpha=1)
cv.out

bestlam = cv.out$lambda.min
bestlam

coef = predict(cv.out, type = 'coefficients', s = bestlam)
coef

coef <- coef(cv.out)
coef

test_x = model.matrix(Y~., test)[,-1] # Prediction
predictions = predict(cv.out, newx = test_x, s = bestlam)
predclass <- ifelse(predictions> 0.5, 1, 0)

table(predclass, test$Y) # Misclassifications

accuracy <- sum(predclass == test$Y) / length(test$Y) # Accuracy
accuracy

# AIC
# Extract coefficients at best lambda
coef <- predict(cv.out, type = "coefficients", s = bestlam)[1,]

# Get non-zero coefficients
non_zero_coef <- coef != 0

# Fit a standard linear model using non-zero coefficients
fit <- lm(y ~ x[, non_zero_coef])

# Calculate AIC
aic <- AIC(fit)
aic

F1_Score(predclass, test$Y) #F1 Score
# ANN----

# Convert character columns to factors
char_cols <- sapply(train, is.character)
train[, char_cols] <- lapply(train[, char_cols], as.factor)

# Convert factor columns to integers
factor_cols <- sapply(train, is.factor)
train[, factor_cols] <- lapply(train[, factor_cols], function(x) as.integer(as.factor(x)))

# Prepare the data
train <- data.frame(train)

# Train the ANN
result = neuralnet(Y ~ destination + passanger + weather + coupon + direction_same + income + gender, train, hidden = c(3:2), linear.output = T)

# Plot the result
plot(result)