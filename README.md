| Model specifications | Train loss | Train accuracy | Test loss | Test accuracy | Training time | Total parameters |
|----------------------|------------|----------------|-----------|---------------|---------------|------------------|
| VGG (1 block) | 0.179 | 97.619 % | 0.508 | 78.261 % | 26.814 s | 896 |
| VGG (2 block) | 0.262 | 88.095 % | 0.599 | 71.739 % | 43.076 s | 19392 |
| VGG (3 blocks) with data augmentation | 0.632 | 63.492 % | 0.664 | 52.174 % | 63.206 s | 93248 |
| VGG16 with transfer learning | 0.000 | 100.000 % | 0.909 | 73.913 % | 350.487 s | 17926209 |
