# DeepChoice: Enhancing E-Commerce with AI Recommendation Systems

![Poster Snapshot](./CS6140Poster_36x24.pptx.pdf)

This repository contains the code, report, and poster for our final project in the CS6140 Machine Learning course at Northeastern University (Spring 2025). The project explores various recommendation system techniques on the Amazon Reviews dataset, including:

- Traditional collaborative filtering methods (SVD, KNN)
- Hybrid approaches (LightFM with WARP loss)
- Deep learning models (MLP with BERT embeddings and one-hot encoded NN)

---

## Project Summary

In today’s AI-powered e-commerce platforms, personalized recommendations are critical to user engagement. We set out to answer:

> Can transformer-based or neural models outperform classic collaborative filtering approaches on large-scale e-commerce data?

Using the 2023 Amazon Reviews dataset from McAuley Lab (32 product categories, 500M+ reviews), we compared five models:

- **SVD** (Singular Value Decomposition)
- **KNN with Means**
- **LightFM** (WARP loss, hybrid CF + metadata)
- **One-Hot Encoded Neural Network**
- **MLP + BERT Embeddings**

---


---

## Dataset

We used the **Amazon Reviews (1996-2023)** dataset curated by [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews). It contains over 500 million reviews across 32 product categories.

Key preprocessing steps included:
- Filtering reviews before 2020
- Dropping users with fewer than 5 reviews
- Train-test splitting via "Leave-Last-Out" strategy

---

## Methods

### Baseline Models
- **SVD** (Latent Factor Model)
- **KNN with Means** (User-based Collaborative Filtering)
- **LightFM** (Hybrid Collaborative + Content Filtering)

### Deep Learning Models
- **One-Hot Encoded Neural Network**
- **MLP with BERT-based Sentence Embeddings** (`all-MiniLM-L6-v2`)

Each model was trained to predict user ratings on unseen items and evaluated using RMSE.

---

## Results Overview

- **SVD** consistently achieved the best RMSE scores, especially in structured product categories (e.g., Office Products).
- **LightFM** generated more diverse recommendations but performed worse on sparse categories.
- **One-Hot NN** provided strong generalization and convergence in Baby Products and Arts & Crafts.
- **MLP + BERT** showed promise in content-rich domains but needed more tuning and data.

### Example RMSE Results:

| Category               | SVD   | KNN   | LightFM | One-Hot NN | MLP |
|------------------------|-------|-------|---------|------------|-----|
| Office Products        | 1.14  | 1.18  | 1.41    | 1.20       | 1.26|
| Arts, Crafts & Sewing  | 1.22  | 1.28  | 1.53    | 1.24       | 1.31|
| Video Games            | 1.49  | 1.55  | 1.45    | 1.51       | 1.60|
| Software               | 1.81  | 1.84  | 1.77    | 1.82       | 1.86|

---

## Key Takeaways

- Collaborative Filtering remains strong in sparse, structured datasets.
- Neural models and LightFM offer better flexibility and diversity in content-rich categories.
- Early stopping was effective — most models converged in 5–6 epochs.
- Transformer-based features have high potential but require more tuning to outperform SVD.

---

## Future Work

- Fine-tuning neural architectures for better generalization
- Exploring **reinforcement learning** to model dynamic user behavior
- Integrating **metadata** and **review sentiment** into deep learning pipelines
- Applying the models to cold-start problems and newer datasets

---

## Technologies Used

- Python 3.10+
- Google Colab with T4 runtime
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- LightFM
- SentenceTransformers (`all-MiniLM-L6-v2`)

---

## References

1. Koren et al. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*.
2. Covington et al. (2016). Deep Neural Networks for YouTube Recommendations. *ACM RecSys*.
3. Zhang et al. (2019). Deep Learning Based Recommender Systems: A Survey. *ACM CSUR*.

---

## Authors

- **Carolina (Yuhan) Li** – li.yuhan5@northeastern.edu  
- **Xuefeng Li** – li.xuefen@northeastern.edu  
- **Yaxin Yu** – yu.yax@northeastern.edu  
- Advisor: Professor **Yifan Hu**

---

## Contact

Feel free to reach out with questions or collaboration ideas.  
This project was presented at the **Northeastern Spring 2025 Student Showcase**.

---

## License

This repository is for academic and educational purposes only.  
Please cite the authors and include the project URL if you reference or use any content.

