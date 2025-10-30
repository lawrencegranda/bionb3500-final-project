# BioNB 3500: Project Proposal

**Cornell University, Fall 2025**  
**Course:** BIONB 3500, NeuroAI  
**Project by:** Lawrence Granda (lg626@cornell.edu)

## Understanding Contextual Meaning in Transformer Embeddings

Transformer-based language models such as BERT and DistilBERT produce contextualized word embeddings. Given an input sentence, these models output a sequence of high-dimensional vectors (one per token) representing how each word’s meaning is shaped by its surrounding context. For example, the word _bank_ appears in very different contexts: _river bank,_ _finance bank,_ or _seating bank_. Although the surface form is identical, we expect the model to internally represent these occurrences differently depending on their meaning.

Our goal is to investigate how these embeddings reflect context-dependent meaning and whether transformer models implicitly separate different senses of the same word. Specifically, we will analyze how the embeddings of ambiguous words distribute in vector space across multiple contexts. We expect embeddings of the same sense (e.g., _river bank_, _muddy bank_) to cluster closely together, while embeddings from different senses (e.g., _bank account_, _river bank_) should be far apart. This leads to our first hypothesis:

1. **H1: Unsupervised Sense Induction**. _If a model truly learns to represent meaning, then the embeddings of the same word used in different contexts should naturally form clusters corresponding to distinct senses, even without explicit supervision._

Building on this, our second hypothesis is:

2. **H2: Layer-wise Semantic Specialization**. _Lower layers of the transformer encode surface and syntactic information, while deeper layers capture more abstract, semantic representations. We hypothesize that sense differentiation emerges and peaks in the middle-to-upper layers, before the final layers become too specialized or task-tuned._

To test these hypotheses, we will extract contextual embeddings for polysemous words such as bank, pitch, and bat from BERT and DistilBERT (or other models of interest, open to suggestions), visualize their structure using dimensionality reduction techniques, and evaluate clustering quality. By comparing models and layers, we aim to understand how architecture size and representational depth influence semantic organization. This work contributes to our understanding of how large language models internalize meaning, and how their internal representations parallel or diverge from human semantic cognition.

To evaluate the feasibility of the proposal, I submitted it to Google Gemini’s Research mode. Here’s the report it generated: [https://gemini.google.com/share/8ef7ab3edcd1](https://gemini.google.com/share/8ef7ab3edcd1). It is important to note that the report may be biased towards accepting the null hypotheses.

These are some related papers testing similar hypotheses:

- (H1 & H2) [Text Clustering with Large Language Model Embeddings](https://doi.org/10.48550/arXiv.2403.15112)
- (H1) [How does BERT capture semantics? A closer look at polysemous words](https://doi.org/10.18653/v1/2020.blackboxnlp-1.15)
- (H2) [Layer-Wise Evolution of Representations in Fine-Tuned Transformers \[...\]](https://doi.org/10.48550/arXiv.2502.16722)
