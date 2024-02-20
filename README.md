![](/static/nature-header.png)
  
> *Human history is a Gaian dream*
# Overview
  
GAIA (Guided Abiogenesis Inquiry Assistant ) is a 400 million parameter conversational language model, that specializes in the field of abiogenesis.
It is created with the purpose of helping with information, related to the origin of life.

# Dataset
The model is trained mostly on English, on over 6244 rows of mixed data with multiple science fields. The dataset consists mainly of of abiogenesis, biology, chemistry and physics knowledge. 
The abiogenesis part is scraped from trusted and reputable sources like Wikipedia, PubMed and ResearchGate. The biology, chemistry and physics portion is 20% each of larger synthetic datasets, provided by [CAMEL AI](https://huggingface.co/camel-ai).
  
# Training information
- Fine-tuned
- Trained for 10 epochs
- Weight decay = 0.001
- Early stopping implemented

# Inference results
GAIA achieved acceptable balance between bias and variance.
