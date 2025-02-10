# Leveraging GPT for ESG Extraction in Cryptocurrency

## Overview
This project explores the use of GPT-4 models for extracting **Environmental, Social, and Governance (ESG)** issues from cryptocurrency-related text data. By leveraging **prompt engineering techniques** and **multi-agent collaboration**, this approach addresses the challenge of **data scarcity** and eliminates the need for fine-tuning models.

## Features
- Utilizes **GPT-4** without fine-tuning for ESG-related text extraction
- Implements various **prompt engineering** techniques:
  - **Zero-shot learning**
  - **Few-shot learning**
  - **Chain-of-Thought (COT)** reasoning
  - **Multi-Agent Debate (MAD)** framework
- Proposes a novel **Multi-Agent Iterative Debate** framework to improve accuracy
- Employs advanced **evaluation metrics** such as:
  - **Cosine Similarity**
  - **Intersection over Union (IoU)**
  - **Precision & Recall**

## Methodology
1. **Data Collection**: Scraped **5,800+ articles** from CoinDesk, a leading cryptocurrency news platform.
2. **Prompt Engineering**: Implemented structured prompt techniques to optimize ESG-related extractions.
3. **Multi-Agent Framework**: Developed a multi-step iterative framework to refine results and improve model accuracy.
4. **Performance Evaluation**: Assessed extraction quality using quantitative metrics such as **semantic similarity** and **token overlap analysis**.

## Key Findings
- **Zero-Shot Learning** outperformed Chain-of-Thought (COT) in direct extraction tasks.
- **COT-based techniques** provided better coherence and generalizability in complex scenarios.
- **Multi-Agent Iterative Debate** enhanced the precision of ESG content extraction by reducing irrelevant outputs.
- **Evaluating extracted text via cosine similarity** proved effective in identifying meaningful ESG-related content.

## Installation & Setup
### Create virtual env
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Create credentials for scrapping
create cred.py in the root directory
```
REDDIT_CLIENT_ID = ''
REDDIT_CLIENT_SECRET = ''
REDDIT_USER_AGENT = ''
REDDIT_USERNAME = ''
REDDIT_PASSWORD = ''
GITHUB_TOKEN = ''
```

## Results & Evaluation
- **Precision & Recall**: Evaluated extracted content against manually annotated data.
- **Intersection over Union (IoU)**: Measured token overlap to ensure completeness of extracted ESG insights.
- **Cosine Similarity**: Applied vector-based semantic similarity to quantify relevance.

## Future Work
- Enhancing **model robustness** by incorporating reinforcement learning feedback loops.
- Expanding the dataset to cover a broader range of **ESG reports and cryptocurrency projects**.
- Applying **retrieval-augmented generation (RAG)** techniques for context-aware ESG classification.
