# Predict_Clicked_Ads_Customer_by_using_Machine_Learning
A model for predicting users who have the potential to click Product Ads.

# Dataset Description
The goal of the project is to Predict who is likely going to click on the Ad on a website based on the features of a user. Following are the features involved in this dataset which is obtained from 
<a href="https://www.rakamin.com/">Rakamin Academy</a>.

| Feature	| Description	| Type |
| :- | :- | :- |
| Unnamed: 0	| ID Customers	| Numeric |
| Daily Time Spent on a Site	| Time spent by the user on a site in minutes. | Numeric |
| Age	| Customer's age in terms of years.	| Numeric |
| Area Income	| Average income of geographical area of consumer.	| Numeric |
| Daily Internet Usage	| Avgerage minutes in a day consumer is on the internet.	| Numeric |
| Male	| Whether or not a consumer was male.	| Categorical |
| Timestamp	| Time at which user clicked on an Ad or the closed window.	| Categorical |
| Clicked on Ad	| No or Yes is indicated clicking on an Ad.	| Target Variable |
| city	| City of the consumer.	| Categorical |
| province	| Province of the consumer.	| Categorical |
| category	| Category of the advertisement.	| Categorical |

# Background
A company in Indonesia wants to know the effectiveness of an advertisement with the method of displaying Ads on their website. The business team at the company wants to optimize its advertising methods on digital platforms to get high-potential users to click on an advertised product. So, after knowing the potential users, they can apply advertising methods so that the costs to be incurred are not too large.

# Objective
This analysis is important because the company is currently working in the field of digital marketing consultant so that it can find out how much achievement the marketed advertisements have before so they can attract customers to view advertisements, with the following goals and objectives:

1. Create a machine learning model that can detect potential users to convert or be interested in an ad. After that, we can optimize the cost of advertising on digital platforms. The algorithm model that will compare the performance results consists of Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Ada Boost, and Gradient Boosting.
2. Determine the algorithm to be used based on the best performance results based on the recall and accuracy evaluation matrix.
3. Analyzing the important factors that determine the user to convert based on the EDA results and the feature importance of the selected algorithm model.
4. Calculate the possible revenue that will be obtained after implementing machine learning.

# Scope of problem
The scope of the problem to solve by measuring the following business matrix:
- Percentage of users who will convert based on expected performance
- Determine factors related to user convert
- Revenue

# Output of ML to be created
Create a machine learning model to predict which users will click on ads on the website.

# Data Analysis

## Univariate

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/num%20histplot.png)
Figure 1. Histogram of Numerical Columns

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/num%20boxplot.png)
Figure 2. Boxplot of Numerical Columns

General observation:
The histogram results show that almost all numeric columns have a slope that is close to symmetrical, there is only 1 numeric variable with a moderate slope, namely 'Regional Income'. This is also consistent with the boxplot results, which is that almost all numeric columns have no outliers except for the 'Area Income' column.

## Bivariate
![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/num%20histplot%20biv.png)
Figure 3. Histogram of Numerical Columns between User Convert and Not Convert

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/num%20boxplot%20biv.png)
Figure 4. Boxplot of Numerical Columns between User Convert and Not Convert

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Age%20vs.%20Daily%20Time%20Spent%20on%20Site.png)

Figure 5. Scatter plot of Age and Daily Time Spent on Site

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Age%20vs.%20Daily%20Internet%20Usage.png)

Figure 6. Scatter plot of Age and Daily Internet Usage

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Daily%20Internet%20Usage%20vs.%20Daily%20Time%20Spent%20on%20Site.png)

Figure 7. Scatter plot of Daily Time Spent on Site and Daily Internet Usage

General observation:
- Customers who clicked advertisements on websites had a median age of 31-40 years.
- The distribution of daily internet usage (in minutes), the potential for users to click on a product is higher for users who rarely use the internet than those who often use the internet.
- Users who don't click on ads also have more Area Income on average than those who click.
- Meanwhile, more of time spent at website in average for users who do not click on ads.
- From the results of the scatterplot between internet use and length of visit to a website, it shows a pattern that is divided into 2 segments, namely active users and non-active users, where active users tend to be less likely to click on ads on websites than non-active users

## Multivariate

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/heatmap.jpg)

Figure 8. Heatmap of All Numerical features

From the correlation results using heatmap, it is not found that there are features that are closely correlated (redundant), so all of the features can be used for modeling. However, by using Pearson correlation, we cannot determine the relationship between features and target variables because the feature target has a categorical type. Therefore, available datasets including categorical variables cannot be used for Pearson correlations. So to reach a relationship all of the features can use PPS (Predictive Power Score) in calculating the relationship between features and their targets

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Correlation_PPScore.png)

Figure 9. PPScore of all Features. 

In the correlation results using PPScore, the features that are quite related to the target (Clicked on Ad) are Daily Internet Usage, Daily Time Spent on Site, Age, Area Income

# Data Preprocessing
## Features to be used in machine learning:
All features used in machine learning include Daily Time Spent on a Site, Age, Area Income, Daily Internet Usage, Male, Timestamp, and Category. In addition, an extraction feature will be carried out in the Timestamp column so that the Month, Day, and Weekday will be used.

## Data Preprocessing:
1. Handle Missing Value in the Daily Time Spent on Site, Area Income, Daily Internet Usage, and Male columns using statistical values such as mean, median, and mode.
2. Feature Extraction in the Timestamp column to create a new column in the form of, day, month, and weekday
3. Feature Encoding uses the One-hot encoding method in the 'Male' column, 'category' using get_dummy, in addition to the target column 'Clicked on Ad' encoding is performed on values where Yes=1 and No=0.
4. Feature Selection. Based on the EDA results, there are no features that are strongly correlated (redundant). Therefore, to create modeling machine learning is used in all features to see the performance results, except the Unnamed: 0 column which contains only unique customer numbers and has no effect on the target. Meanwhile, the 'city' and 'province' columns that have high cardinality values also need to be removed so that there is no dimensional curse. Feature reduction will be carried out if the results of the matrix or performance obtained are not optimal.
5. Handling Outlier Values is done in the 'Area Income' column using the IQR method
6. Split Data by splitting the dataset containing the features used and the target dataset. In doing Split dataset using 70% train proportion and 30% test.

# Modeling
## Machine learning model before normalization/standardization
The first experiment will be modeling with a simple preprocessing stage without standardizing the features to be used for modeling.

Table 1. Experiment 1: Performance results of models before normalization/standardization
![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Experiment%201.PNG)

The performance results of modeling using the initial dataset without normalizing/standardizing are as follows:
- Almost all models of performance result values between test and train do not have too large a gap (overfitting/underfitting) due to using the appropriate parameters.
- From the modeling results on the test, the accuracy value is greater in the AdaBoost and Gradient Boosting models.
- Apart from the accuracy, the recall results on the test also show better on the Gradient Boosting model with the time elapsed it takes the model to predict is also the fastest time than other models, namely 2.00
- For some models such as k-nearest neighbor the resulting accuracy and recall are not good enough.

## Machine learning model after normalization/standardization
Standardize/normalize all features including the target feature using the MinMax Scaler after the data has been split between test and train, this is to ensure that there is no leakage of information about the mean and median to the train data. Scaling is also done on the test data so that we can test and evaluate whether the model can be generalized again.

Table 2. Experiment 2: Performance results of models after normalization/standardization
![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Experiment%202.PNG)

The result of the performance model after applying the Min-Max Scaler to the dataset that has been separated for test and train:
- There is a significant increase in several models, especially for the k-nearest neighbor model.
- There is a change in result best performance, the results of the highest test accuracy are in the Gradient Boosting and AdaBoost models.
- Meanwhile, based on the results of the recall test, the highest was found in the Gradient Boosting and Decision Tree models.
- Based on the experimental results, the second model chosen is Gradient Boosting because it has high accuracy and recall value in addition to the time required to predict is also the fastest among other models.

# Evaluation
## Confusion Matrix
![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Confusion%20Matrix%20GB.png)

Figure 10. Confusion Matrix of Gradient Boosting

From the results of the Confusion Matrix, it is known that the model predicts more users who click on ads because in previous matrix evaluation, in addition to considering the high accuracy value, it also considers recall performance, namely customers who click on ads and can be predicted correctly.
The confusion matrix produced by Gradient Boosting is very good. We can see that there are very few prediction errors in purple cells (top right and bottom left).
With the following results, we will get good accuracy and recall.

## Feature Importance
![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Feature%20Importance%20GB.png)

Figure 11. Feature importance of Gradient Boosting.

By using the Gradient Boosting model we are able to see the most important features in building the model.
Based on the Gradient Boosting method, we can see that daily internet usage is a very important feature in determining whether a user will click or not. Other important features are daily time spent on site, age, and income of an area.

# Business Recommendation
## Feature Based
Based on EDA and feature importance, it can be concluded that:
- The data we get has 2 user segments, namely active and non-active user segments, where active users have the characteristics of users who often use the internet and often visit the website of a product, besides that they have a higher income with an age range of 20-40.
- While non-active users are rarely the opposite of active users.
- Non-active users tend to be more easily attracted to clicking on product ads on digital ads, compared to non-active users.
- Middle old is a potential market for the digital market.
Action points:
- We can change the method of advertising products such as not showing too much advertising so that it can attract the attention of active users.

## Model Based - Simulation
By using the ML model that has been created, we can make the following simulations::

Target Variabel:

![alt text](https://github.com/ayodhyaGA/Predict_Clicked_Ads_Customer_by_using_Machine_Learning/blob/main/fig/Target%20Label.PNG)

1. Before using ML
Assumption:
- To advertise a user can use a budget of Rp.1000
- Using the initial dataset as a simulation implementation with a total of 1000 users, in each class as many as 500 users.
- Every user who converts us will get a profit of Rp.5000
- Without Machine Learning Model

Cost calculation:
Cost = cost ads * n user
Cost = Rp. 1000 * 1000
Cost = Rp. 1.000.000

- While the conversion rate that we will get is 50%
- Because there are only 500 converts, we will get 500 * Rp.5000 = Rp 2.500.000
- Revenue = Rp 2.500.000
- Profit = Rp 2.500.000 - Rp. 1.000.000 = Rp. 1.500.000 
- Based on the simulation above, if we don't use a machine learning model, we will get 1.5 million in revenue

2. After using ML
By Using ML Model Based on the expected performance, we get 95% results on the test results, so when applied to the initial dataset, we will get 950 users who convert based on users with the potential characteristic to click on advertising products.
- With the same cost of ads, which is 1 million
- While the conversion rate that will be obtained is 95% (950 user convert)
- Then we will get 950 * Rp.5000 = Rp.4.750.000
- Revenue= Rp. 4.750.000
- Profit= Rp. 4.750.000 - Rp. 1.000.000  = Rp.3.750.000 
- Based on the simulation above, if we don't use a machine learning model, then we will get revenue 1.5 m and with use ML the revenue increasing significanlty more two times.

**In conclusion, ML can work well into potential revenue**
