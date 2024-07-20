import tensorflow as tf
import numpy as np
import time
from send_to_racing import bbox,depth_values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
# Load the TFLite model
tflite_model_path = 'post_training_quantized_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

X = np.array(bbox)
Y = np.array(depth_values)
df = pd.DataFrame(X , columns= ['class' , 'x ', 'y','w','h', 'confidence'])
df['ratio']  = df['h']/df['w']
#print(df['ratio'])
df = df[['y' , 'w' , 'h' , 'confidence' , 'class' , 'ratio']]#including class here becasue of the mention in ADR(THINK LATER) -will check during feature engineering
X = np.asarray(df).astype('float32')


X_train, X_test, Y_train, Y_test  = train_test_split(X,Y,test_size=0.1, shuffle=True)#X_test for final check .mostly validation will only be used for the model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# Function to run inference
def predict_tflite(interpreter, X):
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Collect predictions
predictions = []
for i in range(len(X_test)):
    X = X_test[i:i+1]
    y_pred = predict_tflite(interpreter, X)
    predictions.append(y_pred[0][0])

# Calculate Mean Squared Error
mse = np.mean((np.array(predictions) - Y_test) ** 2)
print(f'Mean Squared Error: {mse}')

# Measure latency
start_time = time.time()
for i in range(len(X_test)):
    X = X_test[i:i+1]
    _ = predict_tflite(interpreter, X)
end_time = time.time()

total_time = end_time - start_time
average_latency = total_time / len(X_test)
print(f'Average Latency: {average_latency} seconds per sample')
