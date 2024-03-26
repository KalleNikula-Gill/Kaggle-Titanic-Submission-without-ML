import pandas as pd
import numpy as np

def changeAge(age):
    # This is used to replace missing values in the age column 
    if age > 0:
        return age
    # age is only used to seperate <18 and >=18 so the value of 25 doesn't affect anything
    return 25

X = pd.read_csv('/kaggle/input/titanic/train.csv')
y = X.Survived
X['Age'] = [changeAge(age) for age in X['Age']]
X_test = pd.read_csv('/kaggle/input/titanic/test.csv')
X_test['Age'] = [changeAge(age) for age in X_test['Age']]

def getIndividualResult(age,sex,pclass,parch,port):
    if age < 18:
        if pclass == 3 and port != 'C':
            return 0
        return 1
    elif sex == 'female':
        if pclass == 3 and port != 'C':
            return 0
        return 1
    if pclass == 3 and port == 'C' and parch >= 1:
        return 1
    return 0

def getResult(ages,sexes,pclasses,parches,ports):
    result = []
    for i in range(len(ages)):
        result.append(getIndividualResult(ages[i], sexes[i], pclasses[i], parches[i], ports[i]))
    return result

predictions = np.array(getResult(X['Age'], X['Sex'], X['Pclass'], X['Parch'], X['Embarked']))

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(predictions, y)
print("Score:", acc_score) # Score: 0.8092031425364759

final_preds = np.array(getResult(X_test['Age'], X_test['Sex'], X_test['Pclass'], X_test['Parch'], X_test['Embarked']))

output = pd.DataFrame({'PassengerId': X_test.PassengerId,
                       'Survived': final_preds})
output.to_csv('submission.csv', index=False)
