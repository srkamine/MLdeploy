import pickle
# Load the model
modal= pickle.load('diabetic_25.pkl','rb')

# prediction for custom parameter data
data=modal.predict([[1,1,1,1,1,1,1,1]])
print(data)
if data[0] == 0:
    print('Not diabetic')
else:
    print('Diabetic')