Approaches: 
* Full history model
* Window-based regression problem


# Competitions
Forecasts are indispensable for many of the decisions that we make, such as time to get up in the morning in order to not be late for work, or the brand of television to buy that provides the best value for money. Supermarkets require forecasts to support their strategic development, make tactical decisions, and manage their demand and supply planning processes in order to avoid customer service issues and high inventory costs

## M comeptitions
the first three competitions demonstrated the value of combining, the potential of automatic forecasting methods, and the merits of simplicity, 
thw fourth competition showed that machine learning (ML) methods and a hybrid approach utilizing “cross-learning” (Semenoglou et al., 2021) obtained more successful forecasts than the alternatives.
the fivefocusing on a retail sales forecasting application and using real-life, hierarchically structured sales data with intermittent and erratic characteristics
- M1 11001
- M2 1987-1989  (Arima)
- M3 2000 - 3003 series (Famous Theta method )
- M4 2018 - 100,000 time series (different ) 
- M5 2020 - ~42K
- M6

### The M5 Competition (ran from 2 March to 30 June 2020)

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

Companies that support the competition
- [forecastpro](https://www.forecastpro.com/)


# Metrics 
Metrics:

random walk or naıve method (where $f_t+k$ is equal to the last observation)


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

* [M5 simple fe](https://www.kaggle.com/kyakovlev/m5-simple-fe)
* [M5 lags features](https://www.kaggle.com/kyakovlev/m5-lags-features)
* [M5 custom features](https://www.kaggle.com/kyakovlev/m5-custom-features)
* [M5 three shades of dark darker magic](https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic)
* [M5 Forecasting Competition GluonTS Template](https://www.kaggle.com/code/steverab/m5-forecasting-competition-gluonts-template/notebook)
* [Why tweedie works?](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/150614)


# Papers
## Methods
2006
- [x] [Another look at measures of forecast accuracy](https://www.sciencedirect.com/science/article/abs/pii/S0169207006000239): The paper discusses and compares different measures of forecast accuracy, and proposes that the mean absolute scaled error (MASE) be used as the standard measure for comparing forecast accuracy across multiple time series.
2016
- [ ] [TRMF: Temporal regularized matrix factorization for high-dimensional time series prediction](https://dl.acm.org/doi/abs/10.5555/3157096.3157191): A novel temporal regularized matrix factorization (TRMF) framework is proposed for high-dimensional time series analysis. TRMF learns temporal dependencies among latent factors and can be used for forecasting future values.

2017 
- [ ] [DARNN: A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971): stly passes the model inputs through an input attention mechanism and subsequently employs an encoder-decoder model equipped with an additional temporal attention mechanism
- [ ] [A Multi-Horizon Quantile Recurrent Forecaster](https://arxiv.org/abs/1711.11053)
2018
- [ ] [STGCN: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://www.ijcai.org/proceedings/2018/0505.pdf)
- [ ] [**LSTNet**: Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks `SIGIR`](https://arxiv.org/abs/1703.07015)[code](): local multivariate patterns, modeled by a convolutional layer and long-term dependencies, captured by a recurrent network structure
- [ ] [**DeepState**: Deep state space models for time series forecasting `Amazon` `NIPS`](https://papers.nips.cc/paper_files/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html): is a probabilistic generative model that learns to parametrize a linear state space model using RNNs

2019
- [ ] [**DeepGLO**: Think globally, act locally: a deep neural network approach to high-dimensional time series forecasting `Amazon` ](https://dl.acm.org/doi/10.5555/3454287.3454722)[code](https://github.com/rajatsen91/deepglo)
- [ ] [Deep Air Quality Forecasting Using Hybrid Deep Learning Framework](https://ieeexplore.ieee.org/document/8907358): consists of a two-staged feature representation; The data is passed through three 1D convolutional layers, followed by two bi-directional LSTM layers and a subsequent linear layer for prediction
- [ ] [**DeepAR**: Probabilistic forecasting with autoregressive recurrent networks `Amazon` `IJF`](https://www.sciencedirect.com/science/article/pii/S0169207019301888): is an auto-regressive probabilistic RNN model thatestimates parametric distributions from time series with the help of additional time- and categorical covariates
- [ ] [Multi-Horizon Time Series Forecasting with Temporal Attention Learning `KDD`](https://dl.acm.org/doi/10.1145/3292500.3330662)
- [ ] [Tweedie Gradient Boosting for Extremely Unbalanced Zero-inflated Data](https://arxiv.org/abs/1811.10192)
- [ ] [Spatial risk estimation in Tweedie compound Poisson double generalized linear models](https://arxiv.org/abs/1912.12356)

2020
- [ ] [Temporal fusion transformers for interpretable multi-horizon time series forecasting `Google` `IJF`](https://www.sciencedirect.com/science/article/pii/S0169207021000637): ombines recurrent layers for local processing with the transformer-typical self-attention layers that capture long-term dependencies in the data
- [ ] [FORECASTING WITH SKTIME: DESIGNING SKTIME’S NEW FORECASTING API AND APPLYING IT TO REPLICATE AND EXTEND THE M4 STUDY](https://arxiv.org/pdf/2005.08067.pdf)

2021
- [x] [Do We Really Need Deep Learning Models for Time Series Forecasting?](https://arxiv.org/pdf/2101.02118.pdf) - [code](https://github.com/Daniela-Shereen/GBRT-for-TSF)

2022
- [ ] [Forecasting with trees](https://www.sciencedirect.com/science/article/pii/S0169207021001679)
- [ ] [Benchmark time series data sets for PyTorch – the torchtime package](https://arxiv.org/abs/2207.12503)
- [ ] [ETSformer: Exponential Smoothing Transformers for Time-series Forecasting `SaleForce`](https://arxiv.org/abs/2202.01381)
- [ ] [FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://arxiv.org/pdf/2201.12740.pdf)

2023
- [ ] [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://arxiv.org/abs/2304.08424) - [Google Blogs - Recent advances in deep long-horizon forecasting](https://ai.googleblog.com/2023/04/recent-advances-in-deep-long-horizon.html)


## Review / survay

2018
- [ ] [The M4 Competition: Results, findings, conclusion and way forward](https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785)
- [ ] [Considerations of a retail forecasting practitioner](https://www.sciencedirect.com/science/article/abs/pii/S0169207018300293)
- [x] [A classification of business forecasting problems](https://forecasters.org/product/foresight-issue-52/)

2019
- [ ] [Deep Factors for Forecasting](https://arxiv.org/abs/1905.12417)

2020
- [ ] [The M4 Competition: 100,000 time series and 61 forecasting methods](https://www.sciencedirect.com/science/article/pii/S0169207019301128)
- [ ] [Deep Learning for Time Series Forecasting: Tutorial and Literature Survey](https://arxiv.org/abs/2004.10240)
- [ ] [FFORMA: Feature-based forecast model averaging](https://www.sciencedirect.com/science/article/abs/pii/S0169207019300895)

2021
- [ ] [Investigating the accuracy of cross-learning time series forecasting methods `M4`](https://www.sciencedirect.com/science/article/abs/pii/S0169207020301850)
- [ ] [Kaggle forecasting competitions: An overlooked learning opportunity](https://www.sciencedirect.com/science/article/abs/pii/S0169207020301114)
- [ ] [Product sales probabilistic forecasting: An empirical evaluation using the M5 competition data](https://www.sciencedirect.com/science/article/abs/pii/S0925527321002139)

2022
- [x] [M5 accuracy competition: Results, findings, and conclusions `M5`](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [ ] [Exploring the representativeness of the M5 competition data](https://www.sciencedirect.com/science/article/abs/pii/S0169207021001175)
- [ ] [Introduction to the M5 forecasting competition Special Issue](https://www.sciencedirect.com/science/article/pii/S0169207022000565)
- [ ] [Commentary on the M5 forecasting competition](https://www.sciencedirect.com/science/article/abs/pii/S016920702100128X)
- [ ] [M5 competition uncertainty: Overdispersion, distributional forecasting, GAMLSS, and beyond](https://www.sciencedirect.com/science/article/abs/pii/S0169207021001527)
- [ ] [GoodsForecast second-place solution in M5 Uncertainty track: Combining heterogeneous models for a quantile estimation task](https://www.sciencedirect.com/science/article/abs/pii/S0169207022000541)
- [ ] [Forecasting: theory and practice `IJF`](https://www.sciencedirect.com/science/article/pii/S0169207021001758)
- [x] [Criteria for Classifying Forecasting Methods](https://arxiv.org/abs/2212.03523): The paper argues that the distinction between machine learning and statistical forecasting methods is artificial and limits our understanding of the strengths and weaknesses of different forecasting methods.
- [ ] [Transformers in Time Series: A Survey](https://arxiv.org/abs/2202.07125)
- [ ] [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)


## Conferences & Journals
- [International journal of forecasting](https://www.sciencedirect.com/journal/international-journal-of-forecasting) [Alternative](https://forecasters.org/)
- NeurIPS Advances in neural information processing systems.
- KDD
- SIGIR
- SDM
- ECML
- ICML
- [CIKM](http://www.cikmconference.org/)
- IJCAI International Joint Conference on Artificial Intelligence
- ICLR 
- SIGKDD International Conference on Knowledge Discovery & Data Mining.
- 

## Book 
- [ ] [Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3/)



# Code and implementation

* https://github.com/google-research/google-research/tree/master/tft
* https://github.com/awslabs/gluonts
* https://github.com/unit8co/darts
* https://github.com/sktime/sktime-dl
* https://github.com/cuge1995/awesome-time-series#Kaggle-time-series-competition
* https://amlts.github.io/amlts2022/
* https://github.com/kashif/pytorch-transformer-ts
* https://github.com/zalandoresearch/pytorch-ts
* [XGBoostLSS](https://github.com/StatMixedML/XGBoostLSS): a probabilistic XGBoost time series modeling,  XGBoostLSS models all moments of a parametric distribution, i.e., mean, location, scale and shape (LSS), instead of the conditional mean only


# Competitions

- Corporación Favorita Grocery Sales Forecasting
- Recruit Restaurant Visitor Forecasting
- [Jane street market prediction](https://www.kaggle.com/competitions/jane-street-market-prediction/data)


# Blogs

* [Deep learning is what you don't need](https://valeman.medium.com/-86655805a676)

# Video
- [x] [Feature Engineering for Time Series Forecasting - Kishan Manani](https://www.youtube.com/watch?v=2vMNiSeNUjI&ab_channel=DataTalksClub%E2%AC%9B)


# Libraries

- https://github.com/aeon-toolkit/aeon
- 


# Dataset 
type of data:
 - data with seasonality (daily, monthly, yearly, etc)

* [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) : Can you accurately predict sales for a large grocery chain?
* [Walmart Sales Forecasting](https://www.kaggle.com/c/walmart-sales-forecasting/data)
* [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data): Forecast sales using store, promotion, and competitor data
* [Global Energy Forecasting Competition 2012 - Load Forecasting](https://www.kaggle.com/competitions/global-energy-forecasting-competition-2012-load-forecasting/leaderboard): A hierarchical load forecasting problem: backcasting and forecasting hourly loads (in kW) for a US utility with 20 zones.
* [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting): Predict how many future visitors a restaurant will receive

# Related companies
Planning companies 
* https://blueyonder.com




# Categrization
## Classification of Business Forecasting Problems

| Dimensions of the classification | Strategic | Tactical | Operation |
|---|---|---|---|
| Example | Mainly revenue forecasting for strategic decision, such cash flow, topology planning, and market/segment entrance/exit decisions. They are also used to communicate with investors. Other strategic forecasting problems include the trends in long-term energy consumption. | tactical forecasting encompasses promotion planning | demand forecasting in the retail sector as well as short-term energy consumption |
| Forecast Horizon | Long term – many years | Three to six months | days to weeks |
| Time and Product/Location Granularity | highly aggregated | Aggregate the entire brand in a region | Product-store |
| Scale | handful of time series | Few brands | Thousands, even millions, of time series |
| Latency requirements | Long (weeks) | online | hours |
| Consumer of Forecasts | executives or middle management | category managers who negotiate terms with suppliers based on promotion plans and forecasts | automatic systems |
| Characteristics of the Time Series. | regular time series | regular time series | the most difficult time series to forecast because of lumpiness, life cycles, and obsolescence. |
| Drivers | Trend, seasonality, calendar events, macroeconomic variables | Historical promotions and prices, cannibalization, trend and seasonality. | Prices, competitor actions, seasonality, promotional activities, markdowns, ntraweekly seasonality, aggregated store-level data. |
| Form of forecast | point forecasting | point forecasts with some reliance on prediction intervals | The entire forecast distribution |

## Classifying Forecasting Methods
| Category   | Dimension                               |
|------------|-----------------------------------------|
| Objective  | Global vs. Local Methods                |
|            | Probabilistic vs. Point Forecasts       |
|            | Computational Complexity                |
|            | Linearity & Convexity                   |
| Subjective | Data-driven vs. Model-driven            |
|            | Ensemble vs. Single Models              |
|            | Discriminative vs. Generative           |
|            | Statistical Guarantees                  |
|            | Explanatory/Interpretable vs. Predictiv |


