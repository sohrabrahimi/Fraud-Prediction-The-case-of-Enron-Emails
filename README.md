# Fraud-Prediction-The-case-of-Enron-Emails
Project Overview:

In 2000, Enron was one of the largest companies in the United States. 
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal 
investigation, a significant amount of typically confidential information entered into the public record,
including tens of thousands of emails and detailed financial data for top executives. In this project, 
you will play detective, and put your new skills to use by building a person of interest identifier based 
on financial and email data made public as a result of the Enron scandal. To assist you in your detective 
work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which 
means individuals who were indicted, reached a settlement or plea deal with the government, or testified 
in exchange for prosecution immunity.

In this project I will explore the ENron email dataset and examine a number of clasifiers to predictthe POI 
(i.e. Point of Interest) person, based on a number of features extracted from the indivisuals' emails. This 
process will include identifying and removing the outliers, creating new features based on the previous ones, 
and engineering classifiers and tuning them to improve the overall prediction accuracy.

In this project I intended to predict the Points of Interests based on a number of features extracted from the 
Enron email dataset. I took th efollowing steps to accomplish this project: first, I identified a number of outliers
in the dataset and removed them. Next, I created two features and added to the dataset as the new features , which
represented the ratio of emails sent and received from the point of interest to an individual and I found thiese 
features to be more informative. After adding these features, I ended up with a high-dimensional dataset which is
of course prone to overfitting. In order to remedy this, I conducted a PCA and selected only the two first components
and these two components alone explained approximately 99.2 % of the variance in the dataset. Using the first two 
components, I tried a number of different classifiers on this dataset. First, the Gaussian Naive Base model which 
resulted in an overall accuracy of 0.83 (recall:0.84,percision:0.83). This model did not do a good job identifying 
the points of interest and did a very good job in identifying non-POI individuals. 

Next I tried a number of other classifiers and simultanioulsy tuned them using the GridSearchCV() command which enabled
me to easily try a number of parameters for each classifier and choose the best parameters for each. Although the overall
accuracy of SVM is higher than others (i.e. 87 %) this algorithm, same as Naive Basye does a bad job in predicting labels
with 1.0 value. The random forest, however, seems to perfom better in this respect, although the overall accuracay is 84%. 
