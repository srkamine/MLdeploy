import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
url ="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names =['preg','plas','pres','skin','test','mass','pedi','age','class']
df = pd.read_csv(url,names=names)
print(df.head(3))
arr = df.values
x=arr[:,0:8]
y=arr[:,8]
x_train, x_test, y_train,y_test  = model_selection.train_test_split(x,y, test_size=0.2, random_state=101)
modal = LogisticRegression()
modal.fit(x_train, y_train)
result = modal.score(x_test, y_test)
print(result)

# modal saving
filename = 'diabetic_25.pkl'
joblib.dump(modal,filename)
