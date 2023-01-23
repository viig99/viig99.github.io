<!-- +++
title = "Entity Resolution using Contrastive Learning"
description = ""
tags = [
    "entity resolution",
    "ers",
    "ditto",
    "contrastive learning",
    "sentence embeddings"
]
date = "2023-01-01"
categories = [
    "Machine Learning",
    "Entity Resolution",
]
menu = "main"
+++

## **Introduction to Entity Resolution**
[Entity resolution](https://paperswithcode.com/task/entity-resolution) (also known as entity matching, record linkage, or duplicate detection) is the task of finding records that refer to the same real-world entity across different data sources (e.g., data files, books, websites, and databases).

This can be a challenging task, especially when the dataset is large and the queries mention the attributes of the entities in various ways, such as with partial information, typing errors, abbreviations, or extra information. In this blog post, we'll be discussing how to approach the Entity Resolution Problem and the solution that was implemented to solve it.

### **Problem Definition**
Imagine you have a dataset of approximately 50 million entities, and your task is to find the right entity for a given query. The query could be a few of the entity's attributes, and these queries could mention the attributes in various ways. This is the Entity Resolution Problem.

### **The Existing Solution**
One solution to this problem is an Elastic search-based match, which uses complicated heuristics that are overfitted on a small training set. However, this solution is not scalable and the accuracy of the top-20 search retrieval decreases exponentially as the number of entities increases. 

At the time this problem was being addressed, the top-20 search retrieval accuracy was around 40% for the current number of entities.

### **The Implemented Solution**
To solve the Entity Resolution Problem, an embedding search was implemented using a Sentence embedding model. The Deberta model was pretrained and fine-tuned for the current problem using contrastive learning. In contrastive learning, positive pairs are generated using augmentations for each attribute that best mock the queries, based on the many user queries received.

Custom augmentations which syntheically generate query like variations were used during training time to help the model learn generate positive similarity score for entity, query pair.

### **Results**
With this solution, the top-20 accuracy was around 98%. Heuristics and other business logic, along with a properly calculated confidence measure (which was hyperparameter-tuned on the validation set), were used to filter out the right entity. After the final pipeline was implemented, a top-1 accuracy of around 99.995% (precision) and 86% (recall) was achieved for high confidence matches.

In the end, pinecone was chosen for the embedding search and the search latency was around 100ms for the top 50 among the 50 million embeddings.

### **Conclusion**
To conclude, the Entity Resolution Problem was successfully solved by implementing an embedding search using a Sentence embedding model and fine-tuning it with contrastive learning. This solution had a significantly higher accuracy compared to the existing Elastic search-based solution and was able to scale well as the number of entities increased. -->