---
weight: 1
bookFlatSection: false
title: "Work Experience"
---

## Work Experience

{{< columns >}}
### **Staff Machine Learning Engineer**
<--->
### **Kalepa**
<--->
### *Apr 2022 - Sep 2022* - *Toronto, CA*
{{< /columns >}}
- Improved the in-house News Recommendation System using Sentence Embeddings.
- Identified and implemented Document Question Answering, as configurable classifiers to analyse various
risks on businesses.
- Built and deployed Entity Resolution System using [Unsupervised Contrastive Learning](https://arxiv.org/pdf/2202.02098v2.pdf).
    * improved top-20 search accuracy from 35% (on ElasticSearch) to 98% for 30m entities.
    * reduced search latency from 1.5s to 0.3s.
    * built a scalable system using Postgres, Onnx, Pinecone, fastAPI, and Dockerized deployment.
- Interviewed 35+ candidates and developed a system to identify the most suitable candidates for the role.

{{< columns >}}
### **Principal Machine Learning Engineer**
<--->
### **Airtel X-labs**
<--->
### *Sep 2018 - Mar 2022* - *Bangalore, IN*
{{< /columns >}}
- Led the product development of Voicebot engine which powers voice-based queries on the MyAirtel app
with 10m MAU, in 7 indian languages, does 500k queries/day. (Speech to text, Text to speech, training
and inference pipelines.)
    * 900hrs Hindi Speech Dataset Created using [Common Voice](https://github.com/common-voice/common-voice)
    * Used [wav2letter++](https://github.com/flashlight/wav2letter) Streaming Convnets
    * Distributed Training on 16 nodes GPU cluster using OpenMP, RoCE, GPUDirect
    * High performance Bi-directional C++ Grpc Server scaled on k8s
    * Text to Speech built using tactotron2 + vocgan's
    * Voicebot integeration with PBX exchange like Asterix.
- Researched and deployed e2e OCR pipeline serving 1.6m docs/day at 96%+ accuracy, used by Airtel for
its new customer acquisition journey [ICDAR Rank 6](https://bit.ly/35KGMdr "6th Rank on Word Recognition in the wild in ICDAR 2018")
    * Synthetic data creation for Documented Recognition in the Wild.
    * EAST + Convnets as Word Localization & Word Recognition Backbone.
    * Optimized C++ NMS for Zero-copy with pybind11
    * Dynamic parsers DSL based on clustering step.
- Building the workflow-orchestration engine which powers the customer support queries on mail / social
media for Airtel, processes 50k emails/day, built on k8, temporal.io
    * Reverse Engineered and ported workflows for Sprinklr from scratch.
    * Supported 150 different workflows with ~50 activities running concurrently.
    * Maintaining Temporal cluster on OKD, with postgres and cassandra.
- Hired and led a team of 9 engineers.

{{< columns >}}
### **Co-founding Engineer**
<--->
### [**AuthME ID Solutions, Acquired by Airtel**](https://analyticsindiamag.com/airtel-ai-startup-authme/)
<--->
### *Aug 2017 - Sep 2018* - *Bangalore, IN*
{{< /columns >}}
- Built OCR pipeline for reading arbitrary documents, 5 step process with word localization, word
recognition, clustering, parsing and serving.
- Built Voice based IVR bot for Indian business by building on top of DeepSpeech and Rasa NLU.

{{< columns >}}
### **Machine Learning Engineer**
<--->
### **Krowd**
<--->
### *June 2015 - Aug 2017* - *Bangalore, IN*
{{< /columns >}}
- Recommendation & ranking for users by clustering restaurants into latent topics space and finding top-n
using a custom scoring functions built into node.js. Achieved <200ms latency over a set of 1
million restaurants per user. (Fast heuristic approach for cold start case similar to Netflix).
- Loyalty and rewards platform with second price ad bidding for banks (pilot run with Royal Bank of
Scotland).

{{< columns >}}
### **Software Development Engineer**
<--->
### **Amazon**
<--->
### *Feb 2013 - Feb 2015* - *Bangalore, IN*
{{< /columns >}}
- Built the auto correcting & predictive completion language keyboards for regions like germany, japan
etc based on the hidden markov model.
- Worked on Developing & Deploying Amazon Instant Video on 13 different living room TV environments
in 10 months to 1m+ customers.
- Scaling & building a/b testing framework to test the application across various regions.

{{< columns >}}
### **Associate Software Engineer**
<--->
### **Kony Labs**
<--->
### *July 2011 - Nov 2012* - *Hyderabad, IN*
{{< /columns >}}
- Developing an Internal JavaScript single templating based backend/frontend framework for mobile web
development (Single Page Applications).
- Writing native platform level code and integrating it with the existing Lua to native code using Foreign
Function Interface (Objective C, Java).