SVM vs Random Forest?
====================


This project is a Comparison of a support vector machine and a random forest classifier on the same dataset: the BNG_Heart_Statlog data from openml. It is a python built project using jupyter notebooks. The necessary enviornment can be built from the heart_statlog.yml file included. Aim to classify heart disease Absence/0 or Presence/1 

The Process
---------------
We take an initial gander at the dataset and observe 13 features as input, leading to a binary classification: the presence or absence of heart disease. We can see that all of our input data is numerical, but our output data is categorical and of type string.

![Starting Point](presentation/nominal_data.png) 

We can easily fix this problem by using a nominal converter so that we get a numerical and binary output instead. This looks as follows:

![Starting2](presentation/numerical_data.png)

After we've done our initial exploration, we continue by looking at just some visualizations of what we think are important relationships. In this case, we look at the distribution of ages among both classes. Here we can see that, although not a deciding factor, the age distributions of both classes are, in fact, slightly different.

![Starting Point](presentation/init_graphs.png) 
