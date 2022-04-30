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
* Stack for Machine Learning
  *  The Rapid Evolution of the Canonical Stack for Machine Learning - https://opendatascience.com/the-rapid-evolution-of-the-canonical-stack-for-machine-learning/
  *  Navigating the MLOps tooling landscape (Part 1: The Lifecycle) - https://ljvmiranda921.github.io/notebook/2021/05/10/navigating-the-mlops-landscape/
  *  Introducing TWIML’s New ML and AI Solutions Guide - https://twimlai.com/solutions/introducing-twiml-ml-ai-solutions-guide/

## Feature store
* Featureform - https://www.featureform.com/embeddinghub

## Serving
* https://medium.com/@apoor/serving-ml-models-with-grpc-2116cf8374dd

## Drift
* https://medium.com/@elena.samuylova/my-data-drifted-whats-next-how-to-handle-ml-model-drift-in-production-78719ef007b1

## Monitoring and Alerting
* [A comprensive guide od ML monitoring in production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)
* [Monitoring with Boxkite and grafana](https://grafana.com/blog/2021/08/02/how-basisai-uses-grafana-and-prometheus-to-monitor-model-drift-in-machine-learning-workloads/)

## GCP 
- Vertex
  - Model Serving at Scale with Vertex AI : custom container deployment with pre and post processing - https://medium.com/@piyushpandey282/model-serving-at-scale-with-vertex-ai-custom-container-deployment-with-pre-and-post-processing-12ac62f4ce76

- ML Checklist — Best Practices for a Successful Model Deployment - https://medium.com/analytics-vidhya/ml-checklist-best-practices-for-a-successful-model-deployment-2cff5495efed
- Google MLOps template - https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai


# Algorithms / Technique

## NLP 
* Bulk labeling - https://github.com/RasaHQ/rasalit

## Embeddings

- What Are Word Embeddings for Text? - https://machinelearningmastery.com/what-are-word-embeddings/
- An implementation guide to Word2Vec using NumPy and Google Sheets - https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281
- Word2vec from Scratch - https://jaketae.github.io/study/word2vec/
- **Word2Vec Tutorial - The Skip-Gram Model (2016)** - http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
- **The Illustrated Word2vec** - https://jalammar.github.io/illustrated-word2vec/
- **Embeddings with Word2Vec in non-NLP Contexts (Details with papers)** - https://towardsdatascience.com/embeddings-with-word2vec-in-non-nlp-contexts-details-e879b093d34d
- [InferSent](https://medium.com/analytics-vidhya/sentence-embeddings-facebooks-infersent-6ac4a9fc2001)


### Word endedding
- Papers:
  - A Neural Probabilistic Language Model (2003) - https://proceedings.neurips.cc/paper/2000/file/728f206c2a01bf572b5940d7d9a8fa4c-Paper.pdf
  - Efficient Estimation of Word Representations in Vector Space (2013 word2vec) - https://arxiv.org/abs/1301.3781
  - Swivel: Improving Embeddings by Noticing What's Missing (2016 Google) - https://arxiv.org/pdf/1602.02215.pdf

### Sentence Embedding
* [**Universal Sentence Encoder for English** (Google 2018)](https://aclanthology.org/D18-2029.pdf)
* [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data - InferSent (Facebook 2018)](https://arxiv.org/pdf/1705.02364v5.pdf)
* [SentEval: An Evaluation Toolkit for Universal Sentence Representations (2018 Facebook)](https://arxiv.org/pdf/1803.05449.pdf)
* [**Multilingual Universal Sentence Encoder for Semantic Retrieval** (Google 2019)](https://arxiv.org/pdf/1907.04307.pdf)
* [**Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** (2019)](https://arxiv.org/pdf/1908.10084.pdf)
* [**Learning Thematic Similarity Metric Using Triplet Networks** / wikipedia sentences similarity](https://aclanthology.org/P18-2009.pdf)

## Tokenizer
- SentencePiece Tokenizer Demystified (`2021`)- https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15

## Attention
- A Guide to the Encoder-Decoder Model and the Attention Mechanism - https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb
- **Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)** - https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- Attn: Illustrated Attention - https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3
- Papers:
  - Neural machine translation by jointly learning to align and translate - https://arxiv.org/pdf/1508.04025.pdf
  - Effective Approaches to Attention-based Neural Machine Translation - https://arxiv.org/pdf/1409.0473.pdf


## Tansformer
BERT user self--supervice loss call next sentence prediction (NSP)
ALBERT Snetence Order prediciction (SOP) wich clain that model is force to learn mode fine-grain datils
ELECTRA (GAN)
[DistilBert (2019)](https://arxiv.org/pdf/1910.01108.pdf)
[TinyBert (2020)](https://arxiv.org/pdf/1909.10351.pdf)
[MobileBert](https://arxiv.org/pdf/2004.02984.pdf)
Logformer (hybrid local en global attention)

- Curated list of transformer (Dair)- https://github.com/dair-ai/Transformers-Recipe
- **Illustrated transformer**- https://jalammar.github.io/illustrated-transformer/
- Transformers Explained Visually 
  - (Part 1): Overview of Functionality - https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452
  - (Part 3): Multi-head Attention, deep dive - https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
  - (Part 2): How it works, step-by-step - https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34
- Illustrated: Self-Attention - https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
-  https://towardsdatascience.com/galerkin-transformer-a-one-shot-experiment-at-neurips-2021-96efcbaefd3e
- Dive into Deep Learning: Coding Session#5 Attention Mechanism II - https://www.youtube.com/watch?v=rRQcS1O58xk
- The Illustrated Retrieval Transformer - https://jalammar.github.io/illustrated-retrieval-transformer/
- Transformers from Scratch (Brandon Rohrer 2021) - https://e2eml.school/transformers
- Code to train Language model (hugging face)- https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling
- BERT-ology at 100 kmph - https://thenlp.space/blog/bert-ology-at-100-kmph
- Customize transformer models to your domain - https://thenlp.space/blog/customize-transformer-models-to-your-domain
- Papers:
  - **Attention Is All You Need**- https://arxiv.org/pdf/1706.03762.pdf
  - Improving Language Models by Retrieving from Trillions of Tokens (DeepMind’s RETRO (Retrieval-Enhanced TRansfOrmer) Dec 2021) - https://deepmind.com/research/publications/2021/improving-language-models-by-retrieving-from-trillions-of-tokens
  - Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks (2020 ALLEN) - https://arxiv.org/pdf/2004.10964.pdf
  - Natural Language Processing (NLP) for Semantic Search Online Book (pinecone.io) - https://www.pinecone.io/learn/dense-vector-embeddings-nlp/

### BERT
- Explaining BERT Simply Using Sketches - https://mlwhiz.medium.com/explaining-bert-simply-using-sketches-ba30f6f0c8cb
- How to Train a BERT Model From Scratch - https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
- LawBERT: Towards a Legal Domain-Specific BERT? - https://towardsdatascience.com/lawbert-towards-a-legal-domain-specific-bert-716886522b49
- Distillation of BERT-Like Models: The Theory - https://towardsdatascience.com/distillation-of-bert-like-models-the-theory-32e19a02641f

### Distillation
- [**Distillation of BERT-Like Models: The Theory**](https://towardsdatascience.com/distillation-of-bert-like-models-the-theory-32e19a02641f)
- [**Distillation of BERT-like models: the code**](https://towardsdatascience.com/distillation-of-bert-like-models-the-code-73c31e8c2b0a)
- 
### BigBird
- BigBird Research Ep. 1 - Sparse Attention Basics - https://www.youtube.com/watch?v=YvA9nqPmGWg

### Courses
* http://web.stanford.edu/class/cs224n/
* https://www.coursera.org/specializations/natural-language-processing
* https://github.com/dair-ai/ML-YouTube-Courses/blob/main/README.md

## REcSYs
- See more [here](https://github.com/creyesp/Awesome-recsys)

## Reinforment learning
### Next best action
- **NBA** - https://blog.griddynamics.com/building-a-next-best-action-model-using-reinforcement-learning/
  - https://github.com/ikatsov/tensor-house
- **Next-Best-Action Recommendation** https://ambiata.com/blog/2020-09-21-next-best-action-concepts/

## Graph 
* Knowledge Graphs in Natural Language Processing @ ACL 2021 - https://towardsdatascience.com/knowledge-graphs-in-natural-language-processing-acl-2021-6cac04f39761
* Graph ML in 2022: Where Are We Now? - https://towardsdatascience.com/graph-ml-in-2022-where-are-we-now-f7f8242599e0

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

## Constrastive learning (supervised / self-supervised)
Contrastive learning is a self-supervised, task-independent deep learning technique that allows a model to learn about data, even without labels.
* **Understanding Contrastive Learning** - https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607
* Contrastive Representation Learning - https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html
* **Introduction to Dense Text Representations** - https://www.youtube.com/watch?v=t4Gf4LruVZ4&list=PL7kaex1gKh6BDLHEwEeO45wZRDm5QlRil 
  - Global and local structute of vector space
  - Losses: Multiple Negative Ranking Loss (Training with in-batch negative InfoNCE or NTXentloss) / Batch Hard Triplet Loss / Triplet Loss / Contrative loss / CosineSimilarity loss
- The InfoNCE loss in self-supervised learning (deeplearning) - https://crossminds.ai/video/the-infonce-loss-in-self-supervised-learning-606fef0bf43a7f2f827c1583/
* Papers:
  - Big Self-Supervised Models are Strong Semi-Supervised Learners / SimCRLv2 (2020)- https://arxiv.org/pdf/2006.10029.pdf
  - Data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language (2022 Facebook)- https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language

## Others
* [Rethinking Pre-training and Self-training](https://arxiv.org/pdf/2006.06882.pdf)

## Applied ml in the industry (papers)

* https://github.com/eugeneyan/applied-ml

### Producto categorization
* Deep Learning: Product Categorization and Shelving - https://medium.com/walmartglobaltech/deep-learning-product-categorization-and-shelving-630571e81e96
* Semantic Vector Search: Tales from the Trenches - https://medium.com/grensesnittet/semantic-vector-search-tales-from-the-trenches-fa8b61ea3680

### Attribute extractyion in a e-commerce
* [**Learning to Extract Attribute Value from Product via Question Answering: A Multi-task Approach** (2020 Google)](https://dl.acm.org/doi/pdf/10.1145/3394486.3403047)

### Product matching
Papers:
* [Product Matching in eCommerce using deep learning (medium)](https://medium.com/walmartglobaltech/product-matching-in-ecommerce-4f19b6aebaca)
* [Neural Network based Extreme Classification and Similarity Models for Product Matching (Ebay)](https://aclanthology.org/N18-3002.pdf)
* [**BERT-based similarity learning for product matching**](https://aclanthology.org/2020.ecomnlp-1.7.pdf)
* [Deep Entity Matching with Pre-Trained Language Models (Megagon Labs)](https://arxiv.org/pdf/2004.00584.pdf)
* [Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?](https://arxiv.org/pdf/2110.06918.pdf)

### Entity matching
* DeepMatch https://github.com/anhaidgroup/deepmatcher
* DeepER http://www.vldb.org/pvldb/vol11/p1454-ebraheem.pdf
* EMTA https://github.com/brunnurs/entity-matching-transformer
* Auto-EM https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Auto-EM.pdf
* Ditto https://arxiv.org/pdf/2004.00584.pdf

### Foodbert
- http://pic2recipe.csail.mit.edu/
- https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
- https://github.com/chambliss/foodbert
- https://deepnote.notion.site/NLP-in-Notebooks-Competition-6616e415f0a44e5c95982e7bc1cb89dd
- Paper:
  - Exploiting Food Embeddings for Ingredient Substitution - https://www.scitepress.org/Papers/2021/102020/102020.pdf

### item2vec
- **Moving Beyond Meta for Better Product Embeddings (MET)** - https://medium.com/1mgofficial/moving-beyond-meta-better-product-embeddings-for-better-recommendations-fa6dd1578777
- **Item2Vec: Neural Item Embeddings to enhance recommendations** - https://tech.olx.com/item2vec-neural-item-embeddings-to-enhance-recommendations-1fd948a6f293
- Papers:
  - Product recommendation at scale (prod2vec yahoo) - https://dl.acm.org/doi/pdf/10.1145/2783258.2788627
  - item2vec (2016) - https://arxiv.org/pdf/1603.04259.pdf
  - Meta-Prod2Vec - Product Embeddings Using Side-Information for Recommendation (2016)- https://arxiv.org/pdf/1607.07326.pdf
  - Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (2018): https://arxiv.org/pdf/1803.02349.pdf
  - Deep neural network marketplace recommenders in online experiments by Avito - https://arxiv.org/pdf/1809.02130.pdf
  - BERTSCORE: EVALUATING TEXT GENERATION WITH BERT (2019) - https://arxiv.org/pdf/1904.09675.pdf
  

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
