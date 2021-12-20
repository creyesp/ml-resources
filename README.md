# ml-resources

# Python 
* https://towardsdatascience.com/best-practices-for-setting-up-a-python-environment-d4af439846a
* https://towardsdatascience.com/data-scientists-guide-to-efficient-coding-in-python-670c78a7bf79
* https://www.mihaileric.com/posts/setting-up-a-machine-learning-project/
* https://towardsdatascience.com/on-writing-clean-jupyter-notebooks-abdf6c708c75
* https://venthur.de/2021-06-26-python-packaging.html
* https://medium.com/@jessicachenfan/taming-your-python-dictionaries-with-dataclasses-marshmallow-and-desert-388dbffedaec


# MLOPS
* https://medium.com/eliiza-ai/getting-started-with-mlops-d10301cef521
	* Improvement 1: Reproducibility
	* Improvement 2: Modularity
	* Improvement #3: Centralised Caching
	* Improvement #4: Scalability
* https://medium.com/geekculture/enhancing-kubeflow-with-mlflow-8983373d0cac
	* mlflow + kubeflow
* https://winder.ai/how-to-build-a-robust-ml-workflow-with-pachyderm-and-seldon/
* https://gradientflow.com/machine-learning-model-monitoring/
	* Model Monitoring Enables Robust Machine Learning Applications
* https://www.ambiata.com/blog/2020-12-07-mlops-tools/
* https://databaseline.tech/ml-cards/
* https://towardsdatascience.com/mlflow-part-2-deploying-a-tracking-server-to-minikube-a2d6671e6455
* https://medium.com/ibm-data-ai/automate-your-machine-learning-workflow-tasks-using-elyra-and-apache-airflow-adf297adc455
* https://medium.com/everything-full-stack/machine-learning-model-serving-overview-c01a6aa3e823
* https://towardsdatascience.com/how-to-measure-data-quality-815076010b37
* https://huyenchip.com/machine-learning-systems-design/toc.html
* Machine Learning Production Pipeline - https://docs.google.com/presentation/d/1mvmJ1PnCe7lWGmSoL80CjLe7N2QpEwkU8x7l62BawME/edit#slide=id.g7eb0adee5f_0_854

## Serving
* https://medium.com/@apoor/serving-ml-models-with-grpc-2116cf8374dd

## Drift
* https://medium.com/@elena.samuylova/my-data-drifted-whats-next-how-to-handle-ml-model-drift-in-production-78719ef007b1

## GCP 
- Vertex
  - Model Serving at Scale with Vertex AI : custom container deployment with pre and post processing - https://medium.com/@piyushpandey282/model-serving-at-scale-with-vertex-ai-custom-container-deployment-with-pre-and-post-processing-12ac62f4ce76

- ML Checklist — Best Practices for a Successful Model Deployment - https://medium.com/analytics-vidhya/ml-checklist-best-practices-for-a-successful-model-deployment-2cff5495efed


# Algorithms / Technique

## NLP 

### Embeddings

- https://machinelearningmastery.com/what-are-word-embeddings/
- https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281
- Word2vec from Scratch - https://jaketae.github.io/study/word2vec/

### Attention
- A Guide to the Encoder-Decoder Model and the Attention Mechanism - https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb
- **Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)** - https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- Attn: Illustrated Attention - https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3
- Papers:
  - Neural machine translation by jointly learning to align and translate - https://arxiv.org/pdf/1508.04025.pdf
  - Effective Approaches to Attention-based Neural Machine Translation - https://arxiv.org/pdf/1409.0473.pdf


### Tansformer

- **Illustrated transformer**- https://jalammar.github.io/illustrated-transformer/
- Transformers Explained Visually 
  - (Part 1): Overview of Functionality - https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452
  - (Part 3): Multi-head Attention, deep dive - https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
  - (Part 2): How it works, step-by-step - https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34
- Illustrated: Self-Attention - https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
-  https://towardsdatascience.com/galerkin-transformer-a-one-shot-experiment-at-neurips-2021-96efcbaefd3e
- Dive into Deep Learning: Coding Session#5 Attention Mechanism II - https://www.youtube.com/watch?v=rRQcS1O58xk
- Papers:
  - **Attention Is All You Need**- https://arxiv.org/pdf/1706.03762.pdf

#### BERT
- Explaining BERT Simply Using Sketches - https://mlwhiz.medium.com/explaining-bert-simply-using-sketches-ba30f6f0c8cb
- How to Train a BERT Model From Scratch - https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
- LawBERT: Towards a Legal Domain-Specific BERT? - https://towardsdatascience.com/lawbert-towards-a-legal-domain-specific-bert-716886522b49
#### BigBird
- BigBird Research Ep. 1 - Sparse Attention Basics - https://www.youtube.com/watch?v=YvA9nqPmGWg

## REcSYs
* https://medium.com/snipfeed/how-to-implement-deep-generative-models-for-recommender-systems-29110be8971f
* Recommendations as Treatments: Debiasing Learning and Evaluation - http://proceedings.mlr.press/v48/schnabel16.pdf

## Graph 
* Knowledge Graphs in Natural Language Processing @ ACL 2021 - https://towardsdatascience.com/knowledge-graphs-in-natural-language-processing-acl-2021-6cac04f39761

## Time Series
* https://towardsdatascience.com/temporal-convolutional-networks-the-next-revolution-for-time-series-8990af826567
* https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46
* [IJCAI 2021 Tutorial: Modern Aspects of Big Time Series Forecasting](https://www.youtube.com/watch?v=AB3I9pdT46c)
* M4 Forecasting Competition: Introducing a New Hybrid ES-RNN Model (Uber) - https://eng.uber.com/m4-forecasting-competition/
* Interpretable Deep Learning for Time Series Forecasting (Google) - https://ai.googleblog.com/2021/12/interpretable-deep-learning-for-time.html

### Papers:
* **N-BEATS: Neural basis expansion analysis for interpretable time series forecasting** - https://openreview.net/pdf?id=r1ecqn4YwB
* Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting - https://arxiv.org/pdf/1907.00235.pdf
* Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting - https://arxiv.org/pdf/1912.09363.pdf

## Learn to rank
* https://medium.com/swlh/ranknet-factorised-ranknet-lambdarank-explained-implementation-via-tensorflow-2-0-part-i-1e71d8923132
* https://bytes.swiggy.com/learning-to-rank-restaurants-c6a69ba4b330?gi=b000dfdf0130


## ONESHOT 
* https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
* https://medium.datadriveninvestor.com/nlp-in-healthcare-entity-linking-48845a762ed7
* https://bytes.swiggy.com/find-my-food-semantic-embeddings-for-food-search-using-siamese-networks-abb55be0b639  (Michel)
* https://towardsdatascience.com/interpreting-semantic-text-similarity-from-transformer-models-ba1b08e6566c

## Constrative learning
* https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607


## Applied ml in the industry (papers)

* https://github.com/eugeneyan/applied-ml

### Producto categorization
* Deep Learning: Product Categorization and Shelving - https://medium.com/walmartglobaltech/deep-learning-product-categorization-and-shelving-630571e81e96
* Semantic Vector Search: Tales from the Trenches - https://medium.com/grensesnittet/semantic-vector-search-tales-from-the-trenches-fa8b61ea3680

### Foodbert
- http://pic2recipe.csail.mit.edu/
- https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
- https://github.com/chambliss/foodbert
- https://deepnote.notion.site/NLP-in-Notebooks-Competition-6616e415f0a44e5c95982e7bc1cb89dd
- Paper:
  - Exploiting Food Embeddings for Ingredient Substitution - https://www.scitepress.org/Papers/2021/102020/102020.pdf

# XIA
* **Explainability and Auditability in ML: Definitions, Techniques, and Tools** - https://neptune.ai/blog/explainability-auditability-ml-definitions-techniques-tools
* The right way to compute your Shapley Values - https://towardsdatascience.com/the-right-way-to-compute-your-shapley-values-cfea30509254
* **A Brief Overview of Methods to Explain AI (XAI)** - https://towardsdatascience.com/a-brief-overview-of-methods-to-explain-ai-xai-fe0d2a7b05d6


# Education

## MLE certification

* https://sathishvj.medium.com/notes-from-my-google-cloud-professional-machine-learning-engineer-certification-exam-2110998db0f5
* https://towardsdatascience.com/how-i-passed-the-gcp-professional-ml-engineer-certification-47104f40bec5
* https://cloud.google.com/training/machinelearning-ai?skip_cache=true
* https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops
* https://www.tensorflow.org/certificate

### Course
- University of Amsterdam Master - https://uvadlc-notebooks.readthedocs.io/en/latest/index.html
- Dive into Deep Learning - http://d2l.ai/

# Other topics
*  State of AI Report 2021  - https://www.stateof.ai/
* https://medium.datadriveninvestor.com/why-machine-learning-engineers-are-replacing-data-scientists-769d81735553
