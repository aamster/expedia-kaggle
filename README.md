# Hotel Recommendations
In this problem, expedia has asked us to make hotel recommendations for their users. The data consists of 37 million clicks and books by 1.2 million unique users between 2013 to 2014. When a user is ready to book a hotel, we want to be able to predict which hotel the user will book. This will allow expedia to highlight the hotel it thinks the user will book, provide email marketing, and so on. Kaggle sponsored this dataset and problem.

The following models have been implemented:
1) Popularity-based
2) Collaborative filtering
3) Supervised learning

The evaluation metric was MAP@5. The supervised learning method performed significantly better than the others, presumably because it used recency to calculate historical user/hotel interactions, and incorporated additional features.

## Project layout
- metrics/map.py: code for computing performance metric
- /models/model1.py: self-contained code for model 1
- /models/model2.py: self-contained code for model 2
- /models/model3.py: self-contained code for model 3
- grid_search.py: code for iterating over grid of hyperparameters
- Expedia EDA.ipyn: exploratory notebook

## Blog post
https://medium.com/@cosmos1990/building-a-hotel-recommendation-engine-6b9a2b047515

## Code Repository 
https://github.com/aamster/expedia-kaggle
