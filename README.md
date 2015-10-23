# Kaggle Titanic Machine Learning challenge.
The python script provided here is for the machine learning challenge posted on https://www.kaggle.com/c/titanic. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. The goal of this project is to predict the survivers using different features of the passengers on titanic. In the raw data, the features given are the following:

* <font color="red">PassengerId </font> -- A numerical id assigned to each passenger.
* <font color="red">Survived </font> -- Whether the passenger survived (1), or didn't (0).
* <font color="red">Pclass </font> -- The class the passenger was in -- first class (1), second class (2), or third class (3).
* <font color="red">Name </font> -- the name of the passenger.
* <font color="red">Sex </font> -- The gender of the passenger -- male or female.
* <font color="red">Age </font> -- The age of the passenger. Fractional.
* <font color="red">SibSp </font> -- The number of siblings and spouses the passenger had on board.
* <font color="red">Parch </font> -- The number of parents and children the passenger had on board.
* <font color="red">Ticket </font> -- The ticket number of the passenger.
* <font color="red">Fare </font> -- How much the passenger paid for the ticker.
* <font color="red">Cabin </font> -- Which cabin the passenger was in.
* <font color="red">Embarked </font>-- Where the passenger boarded the Titanic.

## summary of the steps taken for this ML task
### organizing data
1. We first fill in the missing values for Age. Some passengers do not have age information, so we replace it with the median age of all passengers.
2. Some passengers have no "Embarked" information. Most passengers embarked from the "S" port, so we replace missing port with "S". We then covert the port to numerical value corresponding to the port.

### engineering new features
1. we can get the total family size as a feature by adding number of siblings/spouse to the number of parents/children on board.
2. The name length is also a feature. People with more prominent status might have longer title/name.
3. The title of a person is a feature. Regex is used to extract the title of a passenger and then assigned to a title id.
4. We assigned each last name to a family id, and uses the family id as a feature. family with less than 3 people are assigned to the id -1, since there are many of them.
5. The result of linear regression is also used as a feature.

### algorithms
Three algorithm are used: logistic regression, random forest, and gradient boosting. The result is determined by a majority vote. The three algorithm each makes a prediciton, and if any two algorithms predict 1, the result is 1; otherwise the result is 0.




