## College Finder by Interest

## Overview

College Finder by Interest is a lightweight AI-powered web application that helps students discover relevant majors and universities based on their interests. By simply entering a sentence like *“I like coding”* or *“I want to become a doctor”*, users receive curated academic recommendations tailored to their input.


## Inspiration

As a high school student, I found it difficult to identify universities that aligned with both my interests and the opportunities they offered. Existing platforms were often too generic or overwhelming. This project was built to simplify that process by providing personalized, interest-based recommendations in an intuitive way.



## Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Model:** TF-IDF (Term Frequency–Inverse Document Frequency)
* **Similarity Measure:** Cosine Similarity


## Model Details

This project uses a TF-IDF + Cosine Similarity approach for text classification.

### How it works:

1. User input is tokenized into words
2. A predefined vocabulary is used to vectorize the text
3. TF-IDF assigns importance to each word
4. Cosine similarity compares input with each category
5. The category with the highest similarity score is selected

This approach is lightweight, interpretable, and effective for small datasets.


## Dataset

A custom dataset was created manually.

### Includes:

* Interest categories (Medicine, Coding, Law, Arts, Music, Space)
* Multiple descriptive phrases per category
* Lists of relevant majors
* Lists of recommended universities

### Example:

* *“software engineering and web development”* - Coding
* *“clinical medicine and patient care”* - medicine

The dataset was optimized to reduce overlap and improve classification accuracy.


## Training Process

This project does not use traditional model training. Instead, it uses:

* Tokenization of dataset and input text
* TF-IDF computation for feature weighting
* Vector representation of text
* Cosine similarity for classification

This makes the system efficient and suitable for real-time browser execution.


## Evaluation Metrics

* **Confidence Score:** Based on cosine similarity
* **Manual Testing:**  Tested across different sentence structures



## Future Improvements

* Top-3 predictions instead of single output
* More advanced NLP models 
  

## Demo




## Acknowledgements

Built as a beginner-friendly AI project to explore real-world applications of NLP and recommendation systems.

