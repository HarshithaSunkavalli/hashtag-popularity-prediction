# hashtag-popularity-prediction
This project is implemented as part of my master thesis in Web and Data Science at Aristotle University of Thessaloniki.  

We gathered twitter data through twitter streaming api and we stored them in a MongoDB collection.  
Steps:  
1) Fetch tweets from database  
2) Calculate hashtag features  
3) Calculate tweet features  
4) Store features in a csv  
5) Read autoencoder data from csv  
6) Use autoencoder as a linear PCA for dimensionality reduction  

## Dependencies can be found on dependencies.txt

![alt text](https://github.com/mpoiitis/hashtag-popularity-prediction/blob/master/Images/15_2_Linear_PCA_AutoEncoder.png)
