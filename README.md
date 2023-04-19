### Performance metrics of different models

| Model specifications | Train loss | Train accuracy | Test loss | Test accuracy | Training time | Total parameters |
|----------------------|------------|----------------|-----------|---------------|---------------|------------------|
| VGG (1 block) | 0.179 | 97.619 % | 0.508 | 78.261 % | 26.814 s | 40,961,153 |
| VGG (2 block) | 0.262 | 88.095 % | 0.599 | 71.739 % | 43.076 s | 20,499,649 |
| VGG (3 blocks) with data augmentation | 0.632 | 63.492 % | 0.664 | 52.174 % | 63.206 s | 10,333,505 |
| VGG16 with transfer learning | 0.000 | 100.000 % | 0.909 | 73.913 % | 350.487 s | 17,926,209 |
| MLP | 0.693 | 51.587 % | 0.693 | 47.826 | 28.423 s | 15,360,257 |  


### Insights
The insights from the observations are as follows:
- The results were not exactly anticipated. The expectation is that number of blocks would result in more overfitting of training data, but that's not what is observed. This can be explained by the fact that at each layer, we are down-sampling the data, making it simpler. Even the number of parameters is reducing. Hence, the observed underftting from VGG1 to VGG3
- Underfitting of data is an issue because the model has to classify Upma and Halwa, which are quite similar dishes. Hence, the model needs to learn training data properly and identify minute differences between the two.
- Data augmentation does not improve performance. This is because more data solves the problem of overfitting. Here the issue is underfitting.  
- Number of epochs do matter for performance. As we observe, the accuracy metrics show a positive trend with incresing the number of epochs. This is because of better fitting of the data, especially when underfitting of data is an issue
- As we expect, the MLP performs significantly worse than other models inspite of having parameters comparable to VGG3. This is because MLPs are not very good with image recognition.