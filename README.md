# Stock Price Prediction using Machine Learning in Python

## Overview

This project explores how machine learning can be used to predict stock trading signals — essentially determining whether buying a particular stock will be profitable. While stock price prediction remains a challenging task due to market volatility and numerous influencing factors, the goal here is to use available data to model short-term price movements for Tesla Inc. stock.

The project uses historical stock price data and applies various ML algorithms to predict a binary signal: whether the stock price will increase the next day. Though the model outcomes highlight the complexity of this task, the project provides a foundational framework for stock trend analysis using machine learning.

---

## Dataset

The dataset consists of Tesla stock prices from January 1, 2010, to December 31, 2017. It includes OHLC (Open, High, Low, Close) data and Volume for each trading day, totaling 1692 rows. The dataset reflects the trading days only, excluding weekends and holidays when markets are closed.

---

## Exploratory Data Analysis (EDA)

- The closing prices show an upward trend over time, indicating strong performance of Tesla stock during the selected period.
- Distribution plots of the OHLC values indicate that the data varies significantly in two distinct regions.
- Volume data is skewed and contains outliers, whereas the OHLC values are fairly clean.
- A strong correlation exists between OHLC features, as expected.
- Prices tend to rise at the end of each financial quarter, suggesting a relationship between quarterly reports and stock performance.

---

## Feature Engineering

Several new features were created to enhance model performance:

- **Day, Month, and Year** were extracted from the date for temporal analysis.
- **is_quarter_end** was added to identify if a date falls at the end of a financial quarter, as these periods often influence stock prices.
- **open-close** and **low-high** differences were calculated to capture daily movement range.
- A **target** variable was created to predict if the stock's closing price on the next day would be higher than the current day.

These engineered features help in gaining deeper insights into stock behavior and improve model learning.

---

## Correlation Analysis

A correlation heatmap confirmed:

- Strong correlation between standard stock price metrics (OHLC).
- Newly engineered features did not introduce redundancy, indicating they can provide unique value to the model.

---

## Data Preparation

- Features were normalized to ensure stable and efficient model training.
- The dataset was split into training and validation sets using a 90/10 ratio.
- The models were evaluated on their ability to predict the probability of a price increase (target = 1).

---

## Model Development and Evaluation

Three models were trained and compared:

1. **Logistic Regression**
2. **Support Vector Classifier (SVC) with polynomial kernel**
3. **XGBoost Classifier**

### Performance Summary:

- **XGBoost Classifier** showed the highest training accuracy (~96%) and best validation performance, though the large gap between training and validation accuracy suggests overfitting.
- **Logistic Regression** demonstrated more consistent results between training and validation, though the overall accuracy was modest (~54% validation ROC-AUC).
- **SVC** performed poorly on both training and validation sets.

A confusion matrix was plotted for the Logistic Regression model to visualize classification outcomes on the validation dataset.

---

## Conclusion

Despite using sophisticated machine learning models and thorough feature engineering, the final results indicate that predicting stock prices with such models may not significantly outperform random guessing (~50% baseline). This highlights the complexity of stock market behavior and suggests that more advanced models, better features, or alternative approaches (like time-series forecasting or deep learning) may be needed.

---

## Tools & Libraries

- **Python**
- **Pandas**, **NumPy** for data manipulation
- **Matplotlib**, **Seaborn** for data visualization
- **Scikit-learn** for model development and evaluation
- **XGBoost** for high-performance gradient boosting
- **Google Colab** as the development environment

---


