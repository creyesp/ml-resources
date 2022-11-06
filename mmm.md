Media Mix Modeling

"The goal of MMM is to understand the drivers of sales, measuring the impact of all factors that may influence sales. These factors can be assigned to two main groups: the group of factors having an only indirect influence on sales (also called baseline) such as economical situation, holidays, weather, competition, and factors that have a direct influence on sales (also called marketing contribution) such as spend on advertising (ad spend) on different media channels like TV, Radio, Online Platforms or price, and promotions."

Main components:
* KPI / target: the variable to predict, ex: sales, adquisitions, installed apps, ...
* media / Channels: click en ad, email, etc
* extra_regresors / control variables: trend, seasonality, holidays, special days, competitors, etc
* cost: 

$$ y_T = \beta_0 + \sum_{m=1}^M \beta_mf(x_{mt}) + \sum_{c=1}^C \beta_c z_{ct} + e_t$$

Feature transformations:
* carryover effect: decaying effects over time (adstock, h)
* diminishing returns: saturation effect between spend and sales

Business metrics:
* CAC: Customer adquisition cost $CAC = x_{mt}/\beta_m f(x_{mt})$
* iCAC: incremental Customer adquisition cost


Papers:
* [**Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects**](https://research.google/pubs/pub46001/)
* [ Challenges and Opportunities in Media Mix Modeling ](https://research.google/pubs/pub45998/)
* [ Geo-level Bayesian Hierarchical Media Mix Modeling ](https://research.google/pubs/pub46000/)
* [ A Hierarchical Bayesian Approach to Improve Media Mix Models Using Category Data ](https://research.google/pubs/pub45999/)
* [ Bias Correction For Paid Search In Media Mix Modeling ](https://research.google/pubs/pub46861/)

Blogs:
* [**How Google LightweightMMM Works**](https://getrecast.com/google-lightweightmmm/)
* [Implementing Uber's Marketing Mix Model With Orbit](https://forecastegy.com/posts/implementing-uber-marketing-mix-model-with-orbit/)
* [**How To Create A Marketing Mix Model With LightweightMMM**](https://forecastegy.com/posts/how-to-create-a-marketing-mix-model-with-lightweightmmm)
* [**Modeling Marketing Mix using PyMC3**](https://towardsdatascience.com/modeling-marketing-mix-using-pymc3-ba18dd9e6e68)
* [Python/STAN Implementation of Multiplicative Marketing Mix Model](https://towardsdatascience.com/python-stan-implementation-of-multiplicative-marketing-mix-model-with-deep-dive-into-adstock-a7320865b334)
* [Carryover and Shape Effects in Media Mix Modeling: Paper Review](https://towardsdatascience.com/carryover-and-shape-effects-in-media-mix-modeling-paper-review-fd699b509e2d)
* [Bayesian Marketing Mix Modeling in Python via PyMC3](https://towardsdatascience.com/bayesian-marketing-mix-modeling-in-python-via-pymc3-7b2071f6001a)
* [A Bayesian Approach to Media Mix Modeling by Michael Johns & Zhenyu Wang](https://discourse.pymc.io/t/a-bayesian-approach-to-media-mix-modeling-by-michael-johns-zhenyu-wang/6024)
* [Upgraded Marketing Mix Modeling in Python](https://towardsdatascience.com/an-upgraded-marketing-mix-modeling-in-python-5ebb3bddc1b6)
* [**Bayesian Media Mix Modeling using PyMC3, for Fun and Profit** `Hellofresh`](https://engineering.hellofresh.com/bayesian-media-mix-modeling-using-pymc3-for-fun-and-profit-2bd4667504e6)
* [Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.io/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
* [Improving the Speed and Accuracy of Bayesian Media Mix Models](https://www.pymc-labs.io/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)
* [Bayesian Media Mix Models: Modelling changes in marketing effectiveness over time](https://www.pymc-labs.io/blog-posts/modelling-changes-marketing-effectiveness-over-time/)
* [Masterclass - Facebook Robyn Tutorial for Marketing Mix Modeling](https://www.youtube.com/playlist?list=PLdaWFt7A-Gf0iyEHwRTuneNN9wKQmJ-QB)
* [Improving Marketing Mix Modeling Using Machine Learning Approaches](https://towardsdatascience.com/improving-marketing-mix-modeling-using-machine-learning-approaches-25ea4cd6994b)
* [Bayesian Marketing Mix Modeling in Python via PyMC3](https://towardsdatascience.com/bayesian-marketing-mix-modeling-in-python-via-pymc3-7b2071f6001a)
