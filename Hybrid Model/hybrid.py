# Hybrid Deep Learning Model

# Part 1 - Identify the Frauds with the Self-Organizing Map
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset= pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling using normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
# Red circle = not approved
# Green square = approved
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o' , 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    
show()

# Finding the frauds
# Indexes of outliers are 2/6 and 1/7 in this example and might differ from
# run to run depending on random initialized weights of the SOM
# Pick the whitest squares from the SOM visualization
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2,6)], mappings[(1,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)



# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Create the matrix of features
customers = dataset.iloc[:, 1:].values

# Create the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if (dataset.iloc[i,0] in frauds):
        is_fraud[i] = 1
        
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# --- Create the ANN ---

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
# output_dim set to avg(dependent variables (1) + independent variables(11))
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =  ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(customers, is_fraud, batch_size = 8, epochs = 100)

# --- Prediction + Model Evaluation ---

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis = 1 ) 

# Rank the predicted frauds
y_pred = y_pred[y_pred[:,1].argsort()]