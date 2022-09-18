# Compressing Feature Space For Classification Using PCA
![image](https://www.rollingstone.com/wp-content/uploads/2018/06/ed-sheeran-cover-story-rolling-stone-interview-8e2ddd75-2223-4cad-ae6a-247b9f0d9e9b.jpg?w=1024)

In this project we use Principal Component Analysis (PCA) to compress 100 unlabelled, sparse features into a more manageable number for classiying buyers of Ed Sheeran’s latest album.
## Overview
### Context
Our client is looking to promote Ed Sheeran’s new album - and want to be both targeted with their customer communications, and as efficient as possible with their marketing budget.

As a proof-of-concept they would like us to build a classification model for customers who purchased Ed’s last album based upon a small sample of listening data they have acquired for some of their customers at that time.

If we can do this successfully, they will look to purchase up-to-date listening data, apply the model, and use the predicted probabilities to promote to customers who are most likely to purchase.

The sample data is short but wide. It contains only 356 customers, but for each, columns that represent the percentage of historical listening time allocated to each of 100 artists. On top of these, the 100 columns do not contain the artist in question, instead being labelled artist1, artist2 etc.

We will need to compress this data into something more manageable for classification!

### Actions
We firstly needed to bring in the required data, both the historical listening sample, and the flag showing which customers purchased Ed Sheeran’s last album. We ensure we split our data a training set & a test set, for classification purposes. For PCA, we ensure that we scale the data so that all features exist on the same scale.

We then apply PCA without any specified number of components - which allows us to examine & plot the percentage of explained variance for every number of components. Based upon this we make a call to limit our dataset to the number of components that make up 75% of the variance of the initial feature set (rather than limiting to a specific number of components). We apply this rule to both our training set (using fit_transform) and our test set (using transform only)

With this new, compressed dataset, we apply a Random Forest Classifier to predict the sales of the album, and we assess the predictive performance!

### Results
Based upon an analysis of variance vs. components - we made a call to keep 75% of the variance of the initial feature set, which meant we dropped the number of features from 100 down to 24.

Using these 24 components, we trained a Random Forest Classifier which able to predict customers that would purchase Ed Sheeran’s last album with a Classification Accuracy of 93%.

### Growth/Next Steps
We only tested one type of classifier here (Random Forest) - it would be worthwhile testing others. We also only used the default classifier hyperparameters - we would want to optimise these.

Here, we selected 24 components based upon the fact this accounted for 75% of the variance of the initial feature set. We would instead look to search for the optimal number of components to use based upon classification accuracy.
## Data Overview
Our dataset contains only 356 customers, but 102 columns.

In the code below, we:

* Import the required python packages & libraries
* Import the data from the database
* Drop the ID column for each customer
* Shuffle the dataset
* Analyse the class balance between album buyers, and non album buyers

![1](https://user-images.githubusercontent.com/100878908/190890251-c71bdd40-87ae-4f25-93fc-44b5afe46a86.png)

From the last step in the above code, we see that 53% of customers in our sample did purchase Ed’s last album, and 47% did not. Since this is evenly balanced, we can most likely rely solely on Classification Accuracy when assessing the performance of the classification model later on.

After these steps, we have a dataset that looks like the below sample (not all columns shown):

![2](https://user-images.githubusercontent.com/100878908/190890345-cb9d6fba-d37c-4966-82bd-4e0be8e7dbcf.png)
## PCA Overview
Principal Component Analysis (PCA) is often used as a Dimensionality Reduction technique that can reduce a large set of variables down to a smaller set, that still contains most of the original information.

In other words, PCA takes a high number of dimensions, or variables and boils them down into a much smaller number of new variables - each of which is called a principal component. These new components are somewhat abstract - they are a blend of some of the original features where the PCA algorithm found they were correlated. By blending the original variables rather than just removing them, the hope is that we still keep much of the key information that was held in the original feature set.

In supervised learning, we often focus on Feature Selection where we look to remove variables that are not deemed to be important in predicting our output. PCA is often used in a similar way, although in this case we aren’t explicitly removing variables - we are simply creating a smaller number of new ones that contain much of the information contained in the original set.

Business consideration of PCA: It is much more difficult to interpret the outputs of a predictive model (for example) that is based upon component values versys the original variables.
## Data Preparation
### Split Out Data For Modelling
In the next code block we do two things, we firstly split our data into an X object which contains only the predictor variables, and a y object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 80% of the data for training, and the remaining 20% for validation. We make sure to add in the stratify parameter to ensure that both our training and test sets have the same proportion of customers who did, and did not, sign up for the delivery club - meaning we can be more confident in our assessment of predictive performance.

![3](https://user-images.githubusercontent.com/100878908/190890558-2a9f505c-84d0-4f7a-a401-d7d482638564.png)

### Feature Scaling

Feature Scaling is extremely important when applying PCA - it means that the algorithm can successfully “judge” the correlations between the variables and effectively create the principal compenents for us. The general consensus is to apply Standardisation rather than Normalisation.

The below code uses the in-built StandardScaler functionality from scikit-learn to apply Standardisation to all of our variables.

![4](https://user-images.githubusercontent.com/100878908/190890563-462907a5-58e5-4312-b15d-fed9f8950d06.png)
## Fitting PCA 
We firstly apply PCA to our training set without limiting the algorithm to any particular number of components, in other words we’re not explicitly reducing the feature space at this point.

Allowing all components to be created here allows us to examine & plot the percentage of explained variance for each, and assess which solution might work best for our task.

In the code below we instantiate our PCA object, and then fit it to our training set.

![5](https://user-images.githubusercontent.com/100878908/190890565-bab4c640-a286-4c76-b775-33662f325ee1.png)
## Analysis Of Explained Variance
There is no right or wrong number of components to use - this is something that we need to decide based upon the scenario we’re working in. We know we want to reduce the number of features, but we need to trade this off with the amount of information we lose.

In the following code, we extract this information from the prior step where we fit the PCA object to our training data. We extract the variance for each component, and we do the same again, but for the cumulative variance. Will will assess & plot both of these in the next step.

![7](https://user-images.githubusercontent.com/100878908/190890836-3d0a44fe-bdad-4f7a-bef0-421cc8877c8e.png)

In the following code, we create two plots - one for the variance of each principal component, and one for the cumulative variance.

![8](https://user-images.githubusercontent.com/100878908/190890839-a7505bbf-e7d2-451e-adad-90fe515b6194.png)

As we can see in the top plot, PCA works in a way where the first component holds the most variance, and each subsequent component holds less and less.

The second plot shows this as a cumulative measure - and we can how many components we would need remain in order to keep any amount of variance from the original feature set.

![9](https://user-images.githubusercontent.com/100878908/190890842-ac3735b4-6f93-443e-97e9-dd8d151f58ca.png)

# Applying our PCA Solution
Now we’ve run our analysis of variance by component - we can apply our PCA solution.

In the code below - we re-instantiate our PCA object, this time specifying that we want the number of components that will keep 75% of the initial variance.

We then apply this solution to both our training set (using fit_transform) and our test set (using transform only).

Finally - based on this 75% threshold, we confirm the number of components that this leaves us with.

![11](https://user-images.githubusercontent.com/100878908/190891046-e11d9506-8949-41bf-9e33-4667548d896b.png)

Turns out we were almost correct from looking at our chart - we will retain 75% of the information from our initial feature set, with only 24 principal components.
## Pickling the PCA model
To use the model for deployment, we saved the trained model in a pickle file.
![10](https://user-images.githubusercontent.com/100878908/190890977-642be501-22d5-4276-b6d1-65e0b2519c05.png)
# Classification Model
## HyperParameter Training to find the best Random Forest Model for Classification
To start with, we will simply apply a Random Forest Classifier to see if it is possible to predict based upon our set of 24 components.
In the code below, we instantiate the grid search to find best parameters for the model.

![12](https://user-images.githubusercontent.com/100878908/190891196-4c17ab84-cca4-49ad-81b6-be44cb595410.png)

Based on grid search, the best parameters were: max_depth = 9 and n_estimators = 500.

## Training the Classifier
We trained the Random Forest Classifier based on Grid search results.
The code the instantiatiang the model is given below:

![13](https://user-images.githubusercontent.com/100878908/190891404-46166e72-556b-40b6-b234-8bb4e1af9110.png)

## Plotting the Confustion matrix and Assesing the Accuracy, Precision, Recall and F1 scores

The confusion matrix:

![14](https://user-images.githubusercontent.com/100878908/190891490-d2bdbdb8-f15c-4502-8178-079492677ef0.png)

Accuracy, Precision, Recall and F1 scores:

![15](https://user-images.githubusercontent.com/100878908/190891535-3cdbf5e8-544c-4059-9db1-fc11ea30d6dd.png)

The result of this is a 93% classification accuracy, in other words, using a classifier trained on 24 principal components we were able to accurately predict which test set customers purchased Ed Sheeran’s last album, with an accuracy of 93%.

## Pickling the Model 
The classification model was saved in a pickle file for the deployment purpose.
![16](https://user-images.githubusercontent.com/100878908/190891653-135cbfe6-6576-4b3c-82cc-c7fcc40fe89b.png)

## Applications
Based upon this proof-of-concept, we could go back to the client and recommend that they purchase some up to date listening data. We would could apply PCA to this, create the components, and predict which customers are likely to buy Ed’s next album.
## Growth & Next Steps
We only tested one type of classifier here (Random Forest) - it would be worthwhile testing others. We also only used the default classifier hyperparameters - we would want to optimise these.

Here, we selected 24 components based upon the fact this accounted for 75% of the variance of the initial feature set. We would instead look to search for the optimal number of components to use based upon classification accuracy.