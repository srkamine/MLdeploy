import joblib
# Load the model
modal= joblib.load('diabetic_25.pkl')

# prediction for custom parameter data
data=modal.predict([[1,1,1,1,1,1,1,1]])
print(data)
if data[0] == 0:
    print('Not diabetic')
else:
    print('Diabetic')