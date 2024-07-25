# In-vehicle-coupon-cat-data-analysis

## Overview

This project explores the dataset collected via a survey on Amazon Mechanical Turk to predict whether a driver will accept a coupon based on various driving scenarios. The dataset and related research are detailed in the paper by Wang, Tong, Cynthia Rudin, Finale Doshi-Velez, Yimin Liu, Erica Klampfl, and Perry MacNeille titled "A Bayesian Framework for Learning Rule Sets for Interpretable Classification," published in *The Journal of Machine Learning Research*.

## Dataset Information

- **Number of Variables**: 26
- **Number of Observations**: 12,684
- **Data Types**:
  - Character: 18
  - Numeric: 8

### Variables and Descriptions

- **destination**: Destination of the user (categorical)
- **passenger**: Passenger in the car (categorical)
- **weather**: Weather during driving (categorical)
- **temperature**: Temperature in Fahrenheit (integer)
- **time**: Time of driving (categorical)
- **coupon**: Type of coupon offered (categorical)
- **expiration**: Coupon expiration time (categorical)
- **gender**: User's gender (categorical)
- **age**: User's age (categorical)
- **maritalStatus**: User's marital status (categorical)
- **has_children**: If the user has children (integer)
- **education**: User's educational level (categorical)
- **occupation**: User's occupation (categorical)
- **income**: User's income level (categorical)
- **car**: Type of car driven by the user (categorical)
- **Bar**: Frequency of visiting bars (categorical)
- **CoffeeHouse**: Frequency of visiting coffee houses (categorical)
- **CarryAway**: Frequency of getting take-away food (categorical)
- **RestaurantLessThan20**: Frequency of visiting restaurants with an average expense below $20 (categorical)
- **Restaurant20To50**: Frequency of visiting restaurants with an average expense between $20 and $50 (categorical)
- **toCoupon_GEQ5min**: Driving distance to use the coupon is greater than 5 mins (integer)
- **toCoupon_GEQ15min**: Driving distance to use the coupon is greater than 15 mins (integer)
- **toCoupon_GEQ25min**: Driving distance to use the coupon is greater than 25 mins (integer)
- **direction_same**: Distance to the coupon location is in the same direction as the user's current destination (integer)
- **direction_opp**: Distance to the coupon location is in the opposite direction of the user's current destination (integer)
- **Y**: Whether the coupon is accepted (1) or rejected (0) (integer)

## Project Team

- **LAU Ngai Sang**
- **Chan Tin Yan**
- **Lam Gloria Kit Sum**
- **Wong Sin Kan**
- **Chan Shun Yin**

## Methodology

### Pre-Processing

- **Missing Data Handling**:
  - Removed columns with a high percentage of missing values.
  - Replaced missing values with the most frequent value to preserve the original data distribution.
  
### Models Used

1. **Logistic Regression**
   - Full Model
   - Interaction Terms
   - LASSO (Least Absolute Shrinkage and Selection Operator)
2. **Artificial Neural Network (ANN)**
   - Included 7 selected variables
   - Structure: 5 x 3 hidden layers

### Formulas

#### Logistic Regression
The logistic regression model is defined as:

\[ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n)}} \]

Where:
- \( P(Y=1|X) \) is the probability of the coupon being accepted.
- \( \beta_0, \beta_1, ..., \beta_n \) are the coefficients of the model.
- \( X_1, X_2, ..., X_n \) are the predictor variables.

#### Interaction Terms
To include interaction terms between two variables, for instance between Bar Coupon and Bar Frequency, the model is:

\[ \text{Interaction Term} = \beta_3 (\text{Bar Coupon} \times \text{Bar Frequency}) \]

#### LASSO Regression
LASSO modifies the cost function of the linear regression to include the absolute value of the magnitude of the coefficients:

\[ \text{Cost Function} = \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \]

Where:
- \( \lambda \) is the penalty parameter.

#### Artificial Neural Network (ANN)
For the ANN, the forward propagation for one hidden layer is given by:

\[ Z^{[1]} = W^{[1]}X + b^{[1]} \]
\[ A^{[1]} = g(Z^{[1]}) \]
\[ Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \]
\[ \hat{Y} = g(Z^{[2]}) \]

Where:
- \( W \) and \( b \) are the weights and biases.
- \( g \) is the activation function.

### Performance Metrics

- **Logistic Regression (Full Model)**
  - Accuracy: 69.12%
  - AIC: 11502
  - F1 Score: 0.6154

- **Logistic Regression (Interaction Terms)**
  - Accuracy: 68.46%
  - AIC: 12260
  - F1 Score: 0.5845

- **Logistic Regression (LASSO)**
  - AIC: 15861
  - F1 Score: 0.6065

- **Artificial Neural Network**
  - Accuracy: 67.67%
  - AIC: 13325
  - F1 Score: 0.6094
![image](https://github.com/user-attachments/assets/e53ed775-401d-45c4-a808-9bf3ada78efe)

## Conclusion

- **Best Model**: Logistic Regression (Full Model)
- **Worst Model**: Logistic Regression (LASSO Model)
- **Suggestions for Improvement**: 
  - Standardization
  - Feature Selection

## References

- Wang, Tong, Cynthia Rudin, Finale Doshi-Velez, Yimin Liu, Erica Klampfl, and Perry MacNeille. "A Bayesian framework for learning rule sets for interpretable classification." *The Journal of Machine Learning Research* 18, no. 1 (2017): 2357-2393.
