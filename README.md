SVM vs Random Forest?
====================


This project is a Comparison of a support vector machine and a random forest classifier on the same dataset: the BNG_Heart_Statlog data from openml, containing one million entries  of 13 features and a 14th "class" column. It is a python built project using jupyter notebooks. The necessary enviornment can be built from the heart_statlog.yml file included. Aim to classify heart disease Absence/0 or Presence/1 

The Process
---------------
#### Initial Visualizations
We take an initial gander at the dataset and observe 13 features as input, leading to a binary classification: the presence or absence of heart disease. We can see that all of our input data is numerical, but our output data is categorical and of type string.

![Starting Point](presentation/nominal_data.png) 

We can easily fix this problem by using a nominal converter so that we get a numerical and binary output instead. This looks as follows:

![Starting2](presentation/numerical_data.png)

After we've done our initial exploration, we continue by looking at just some visualizations of what we think are important relationships. In this case, we look at the distribution of ages among both classes. Here we can see that, although not a deciding factor, the age distributions of both classes are, in fact, slightly different. We can also see that the spreads of resting heart rates are similar among both classes.

![Starting Point](presentation/init_graphs.png) 

#### Model Selection
After this, I decided on the models that I wanted to compare for this particular dataset. Even though neural networks have had remarkable success classifying similar data, I wanted to try machine learning models that I was unfamiliar with, and so I chose to look at SVM's and Random Forests. 
- SVM's because they have the ability to learn nonlinear relationships between features using kernel methods and for the same reason, we can test it earlier without having to preprocess our data excessively.
- Random Forests because if the relationships are not of high order, the model will perform better, train faster and scale to a larger subset of the data with more accuracy.

#### Model Training
As mentioned previously, we could use the SVM 'kernel trick' to predict with almost no data preparation. And we can see below that after only a only a few tuned hyperparameters. The SVM already classifies the small subset with high accuracy. Below is the confusion matrix with details on the particular test.

![Starting Point](presentation/1.png) 


