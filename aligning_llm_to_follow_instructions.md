Repos:
- [PaLM RLHF](https://github.com/lucidrains/PaLM-rlhf-pytorch)
- [RL4LMs](https://github.com/allenai/RL4LMs)
- [trl](https://github.com/lvwerra/trl/)
- [trlx](https://github.com/CarperAI/trlx)
- [Open-Assistant](https://github.com/LAION-AI/Open-Assistant)


Papers:
- [Fine-Tuning Language Models from Human Preferences `OpenAI`](https://arxiv.org/pdf/1909.08593.pdf) `18 Sept 2019` 
- [LoRA: Low-Rank Adaptation of Large Language Models `Microsoft`](https://arxiv.org/abs/2106.09685) `17 June 2021`
- [Finetuned Language Models Are Zero-Shot Learners (FLAN) `Google`](https://arxiv.org/abs/2109.01652) `3 sept 2021`
- [Multitask Prompted Training Enables Zero-Shot Task Generalization (T0)](https://arxiv.org/abs/2110.08207) `15 oct 2021`
- [Training language models to follow instructions with human feedback (InstructGPT)`OpenAI`](https://arxiv.org/abs/2203.02155) `4 Mar 2022`
- [Finetuned Language Models Are Zero-shot Learners `Google`](https://arxiv.org/pdf/2109.01652.pdf) `8 Feb 2022`
- [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)


Dataset Q&A
- [ELI5](https://huggingface.co/datasets/eli5)
- [ChatGPT prompts](https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts)
- [Human ChatGPT Comparison Corpus (HC3)](https://huggingface.co/datasets/Hello-SimpleAI/HC3)


Blogs:
RLHF
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
- [Creating Your Own ChatGPT: A Guide to Fine-Tuning LLMs with LoRA](https://ai.plainenglish.io/creating-your-own-chatgpt-a-guide-to-fine-tuning-llms-with-lora-d7817b77fac0)

Videos:
- [Reinforcement Learning from Human Feedback: From Zero to chatGPT `HF`](https://www.youtube.com/watch?v=2MBJOuVq380)
- [How ChatGPT works - From Transformers to Reinforcement Learning with Human Feedback (RLHF)](https://www.youtube.com/watch?v=wA8rjKueB3Q)

Fine tune GPT-3
- [How We Build The Highest Confidence GPT-3 Chatbots Available In 2022](https://www.width.ai/post/gpt-3-chatbots): Prompt Optimization & Variable Handling, fill in-context
- [Fine Tuning GPT-3: Building a Custom Q&A Bot Using Embeddings](https://www.mlq.ai/fine-tuning-gpt-3-question-answer-bot/): it use document embedding to find the context to a question and then use the completaion model to get de answer
- [Fine-tuning GPT-3 Using Python to Create a Virtual Mental Health Assistant Bot](https://betterprogramming.pub/how-to-finetune-gpt-3-finetuning-our-virtual-mental-health-assistant-641c1f3b1ef3): fine tine a gpt3 model witn domain specic text and the use promp engine to generate the answer
- [Prompt Engineering Overview `Dair-AI`](https://www.youtube.com/watch?v=dOxUroR57xs): very complete techniques and resources to make promt engineering with SOTA resources (notebook, packages, papers, etc)
- [](https://platform.openai.com/docs/tutorials/web-qa-embeddings)


OpenAI API:
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

input: 2048 tokens
~ 1,000 tokens is about 750 words.
8k examples
150 - 200 examples for fine tune gpt3 

| Model | train cost| inference cost |
|--|--|--|
| Ada			 | $0.0004 / 1K tokens 	| $0.0016 / 1K tokens | 
| Babbage		 | $0.0006 / 1K tokens 	| $0.0024 / 1K tokens | 
| Curie		 | $0.0030 / 1K tokens 	| $0.0120 / 1K tokens | 
| Davinci		 | $0.0300 / 1K tokens 	| $0.1200 / 1K tokens | 


Fine tune GPT-3:
- Zero-shot learning: no demostration are given, only natural language instruction describing the task. 
- One-shot learning: only one demostration is give, in addition natural language instruction description if the task. 
- Few-shot learning: K-examples of context and completition, and then one final example of the context, with the model expected to provide a completition


Vector datase
- Pinecone
- Weaviate
