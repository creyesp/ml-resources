
# Competitions
Forecasts are indispensable for many of the decisions that we make, such as time to get up in the morning in order to not be late for work, or the brand of television to buy that provides the best value for money. Supermarkets require forecasts to support their strategic development, make tactical decisions, and manage their demand and supply planning processes in order to avoid customer service issues and high inventory costs

The M5 Competition (ran from 2 March to 30 June 2020)

It differed from the previous four ones in six important ways, some of which were suggested by the discussants of the M4 Competition.

* It used hierarchical sales data, generously made available by Walmart,
  * item level (3,049 products)
  * product categories level (3 product categories (Hobbies, Foods, and Household) and 7 product departments)
  * stores level (ten stores)
  * departmeent level (located in three States (CA, TX, and WI))


* Besides the time series data, it also included explanatory variables that affect the forecast
  * Mprice
  * promotions
  * day of the week
  * special events (e.g. Super Bowl, Valentine’s Day, and Orthodox Easter)

* The distribution of uncertainty was assessed by asking participants to provide information on four indicative prediction intervals and the median.

* The majority of the more than 42,840 time series display intermittency (sporadic sales including zeros).

* Instead of a single competition to estimate both the point forecasts and the uncertainty distribution, there were two parallel tracks using the same dataset, the first requiring 28 days ahead point forecasts and the second 28 days ahead probabilistic forecasts for the median and four prediction intervals (50%, 67%, 95%, and 99%).

* For the first time, it focused on series that display intermittency, i.e., sporadic demand including zeros.




The historical data range from 2011-01-29 to 2016-06-19. Thus, the products have a (maximum) selling history of 1,9411 days / 5.4 years (test data of h=28 days not included). 


Model benchmarch
Statistical Benchmarks
* Naive
* Seasonal Naive (sNaive)
* Simple Exponential Smoothing1 (SES)
* Moving Averages (MA)
* Croston’s method1 (CRO)
* Optimized Croston’s method (optCro)
* Syntetos-Boylan Approximation (SBA)
* Teunter-Syntetos-Babai method (TSB)
* Aggregate-Disaggregate Intermittent Demand Approach (ADIDA)
* Intermittent Multiple Aggregation Prediction Algorithm1 (iMAPA)
* Exponential Smoothing1 - Top-Down (ES_td)
* Exponential Smoothing – Bottom-Up (ES_bu)
* Exponential Smoothing with eXplanatory variables (ESX)
* AutoRegressive Integrated Moving Average1 - Top-Down (ARIMA_td):
* AutoRegressive Integrated Moving Average – Bottom-Up (ARIMA_bu):
* AutoRegressive Integrated Moving Average with eXplanatory variables (ARIMAX):

* Machine Learning Benchmarks
* Multi-Layer Perceptron (MLP)
* Random Forest (RF)
* Global Multi-Layer Perceptron (GMLP):
* Global Random Forest (GRF)


* [M5 competition by M Open Forecasting Center (MOFC)](https://mofc.unic.ac.cy/m5-competition/)
* [Github of competition](https://github.com/Mcompetitions)
* [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview)
* [International Institude of forcasting](https://forecasters.org/)
* [International journal of forecasting](https://www.sciencedirect.com/journal/international-journal-of-forecasting)



Companies that support the competition
- [forecastpro](https://www.forecastpro.com/)



ToDO:
- [ ] Read all solution available in github
- [ ] Read dissuctions in kaggle





# Metrics 
Metrics:

random walk or “na ̈ıve” method (where Ft is equal to the last observation)


Scale-dependent measures
* MSE
* RMSE
* MAE
* MdAE

Scale independet (Measures based on percentage errors): disasbantage: are sensitive to endefined values when obs data have zeros because by definition: 100 * e_t / Y_t, is T_t = 0 then PE = oo or vary skew distribution when data is close to zero (ex. producto with intermitance). A further disasvantage is thar it asume meaninful zero (not useful for ).

* The MAPE and MdAPE also have the disadvantage that they put a heavier penalty on positive errors than on negative error
* , the value of 2|Yt − Ft|/(Yt + Ft) has a heavier penalty when forecasts are low compared to when forecasts are hig

* MAPE Mean Absolute Percentage Error -> median(100|Yt − Ft|/(Yt))
* MdAPE Median Absolute Percentage Error -> median(100|Yt − Ft|/(Yt))
* sMAPE Symmetric Mean Absolute Percentage Error -> mean(200|Yt − Ft|/(Yt + Ft))
* sMdAPE Symmetric Median Absolute Percentage Error -> median(200|Yt − Ft|/(Yt + Ft))

Measures based on relative errors
* rt = et/e∗t denote the relative error where e∗t is the forecasterror obtained from the benchmark method.
* A serious deficiency in relative error measures is that e∗t can be small.

* MRAE Mean Relative Absolute Error -> mean(|rt|)
* MdRAE Median Relative Absolute Error -> median(|rt|)
* GMRAE Geometric Mean Relative Absolute Error -> gmean(|rt|)

Relative measures
Are interpretable againt the baseline

RelMAE = MAE/MAE_baseline 

Scaled errors
Relative measures and measures based on relative errors both try to remove the scale of the data by comparing the forecasts with those obtained from some benchmark forecast method, usually the na ̈ıve method. 

* Of these measures, we prefer MASE as it is less sensitive to outliers and more easily interpreted than RMSSE, and less variable on small samples than MdASE. If the RMSSE is used, it may be preferable to use the in-sample RMSE from the na ̈ıve method in the denominator of qt.
*  scaled errors should become the standard approach in comparing forecast accuracy across series on different scales

$$ q = e_t{ 1 \over n-1 } sum_{i=2}^n{Y_i - Y_{i-1}} $$


* **MASE** Mean Absolute Scaled Error -> mean(q)  [Paper](https://robjhyndman.com/papers/mase.pdf)

$$ RSMSSE = sqrt{{ 1 \over h } { sum_{t=n+1}^{n+h}{(Y_t - \hat{Y}_t)^2 \over { 1 \over n-1 sum_{i=2}^n{(Y_i - Y_{i-1}})^2}}} $$
* RSMSSE


Of course, there will be situations where some of the existing measures may still be preferred. For example, if all series are on the same scale, then the MAE may be preferred because it is simpler to explain. If all data are positive and much greater than zero, the MAPE may still be preferred for reasons of simplicity. However, in situations where there are very different scales including data which are close to zero or negative, we suggest the MASE is the best available measure of forecast accuracy.


# Papers
Papers:
- [ ] [Introduction to the M5 forecasting competition Special Issue](https://www.sciencedirect.com/science/article/pii/S0169207022000565)
- [ ] [M5 accuracy competition: Results, findings, and conclusions](https://www.sciencedirect.com/science/article/pii/S0169207021001874)

# Blogs
* [Deep learning is what you don't need](https://valeman.medium.com/-86655805a676)

# Libraries
- https://github.com/aeon-toolkit/aeon
- 
