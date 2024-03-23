![](/static/nature-header.png)
  
> *Human history is a Gaian dream*
# Overview
  
GAIA (Guided Abiogenesis Inquiry Assistant ) is a 400 million parameter conversational language model, that specializes in the field of abiogenesis.
It is created with the purpose of helping with information, related to the origin of life.

# Dataset
The model is trained mostly on English, on over 6244 rows of mixed data with multiple science fields. The dataset consists mainly of of abiogenesis, biology, chemistry and physics knowledge. 
The abiogenesis part is scraped from trusted and reputable sources like Wikipedia, PubMed and ResearchGate. The biology, chemistry and physics portion is 33% each of larger synthetic datasets, provided by [CAMEL AI](https://huggingface.co/camel-ai).
  
# Retreival Augmented Generation
The project uses RAG, by retreiving short chunks (respecting the max prompt length of the base model) from document ("The vital question" by Nick Lane), thus providing additional knowledge on inference, outside of the training data.
The retreival component returns the chunk with the highest score, encodes it and joins it with the prompt tensor.

# Training information
- Fine-tuned
- Trained for 50 epochs
- Early stopping implemented

# Inference results
GAIA achieved acceptable balance between bias and variance.

****

_USER: Who are you and how can you help me?_

_GAIA: I am GAIA, the digital entity designed to aid in the exploration of life's origins._

****

_USER: What is the role of CDK in the regulation of the cell during mitosis?_

_GAIA: CDK is a critical cellular component of the cytosine base in living cells. It is involved in the formation of a membrane in the presence of a strong acid, such as sulfuric acid (H2SO4) or phosphorus pentafluoride (H3PO4). When it loses a proton (H+) from the nucleus, it forms a peptide ion, which is then dissociated into its respective ions_

****

_USER: Can you explain to me Panspermia_

_GAIA: Pansermia is an autosomal recessive genetic disorder caused by a mutation in the DNA of an organism. It affects the formation and functioning of proteins, leading to various cellular processes, including cell death, apoptosis, and post-transmission. The disease is caused by mutations in genes located at the ends of chromosomes that protect them from degradation, fusion, and recombination during early embryonic development._

****

_USER:  Explain to me the chemiosmotic gradient_

_GAIA: The chemiosmootic gradiency of a molecule or ion can be represented as:
\- Phosphorus (Phenyl) = Σ(-OH) + (2-methylbutane)
Where ΔH is the change in electronegativity of the ion._

# Prerequisites to run
- python >= 3.x
- pytorch >= 2.1.2
- transformers >= 4.36.2
- langchain >= 0.1.11
- chromadb
- sentence_transformers

# Run
1. Set the preferred inference parameters in config.json
2. `python GAIA400m.py`
