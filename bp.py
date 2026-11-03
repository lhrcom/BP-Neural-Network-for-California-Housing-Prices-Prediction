from turtle import forward
import numpy as np
from numpy.random import bit_generator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def Standardization(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    X_std = (x-mean) / (std+1e-8)
    return X_std,mean,std
    
class Bp():
    def __init__(self,layers,lr=0.005,epochs=10):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []
       
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x*(1-x)
    
    def initialize_weights(self):
        # len(self.layers)
        for i in range(len(self.layers)-1):
            w = np.random.randn(self.layers[i],self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1,self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    # len(self.weights)=3
    def forward(self,X):
        activations = [X]
        # print(activations)
        z_values = []
        for i in range(len(self.weights)):
            z = np.dot(activations[-1],self.weights[i])+self.biases[i]
            z_values.append(z)
            if i < len(self.weights) - 1:
                activations.append(self.sigmoid(z))
            else:
                activations.append(z)
            
            # activations.append(self.sigmoid(z))
            
        return activations,z_values
    
    def backward(self,activations,z_values,y_train):
        w_g = []
        b_g = []
        m = y_train.shape[0]
        delta = activations[-1]-y_train.to_numpy().reshape(-1,1)

        for i in range(len(self.weights) - 1, -1, -1):
            
            dw = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            w_g.insert(0, dw)
            b_g.insert(0, db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
        
        return w_g, b_g

    

    def fit(self,X_train,y_train,verbose=True):
        if not self.weights:
            self.layers[0] = X_train.shape[1]
            self.initialize_weights()
        
        losses = []
        for epoch in range(self.epochs):
            activations,z_values = self.forward(X_train)
            loss = np.mean((activations[-1]-y_train.to_numpy().reshape(-1,1))**2)
            losses.append(loss)

            w_g,b_g = self.backward(activations,z_values,y_train)

            for i in range(len(self.weights)):
                self.weights[i] -= self.lr*w_g[i]
                self.biases[i] -= self.lr*b_g[i]
            
            if verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss:.6f}')
        
        self.losses = losses
        return self
    
    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]



if __name__ == "__main__":
    df = pd.read_excel(r'/Users/micheal/Documents/Datasets/fetch_california_housing.xlsx')
    # print(df.head())

    X = df.drop('target',axis=1)
    y = df['target']
    # print(X.shape[1])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train_std,mean,std = Standardization(X_train)
    # X_test_std,_,_ = Standardization(X_test) 
    X_test_std = (X_test - mean) / (std + 1e-8)
    y_train_std, y_mean, y_std = Standardization(y_train)
    

    # layers = [X_train_std.shape[-1],32,8,1]
    layers = [X_train_std.shape[-1],32,16,8,4,2,1]
    # print(y_train)
    # print(len(layers))
    print(type(X_train_std))
    model = Bp(layers,lr=0.4,epochs=2000)
    model.fit(X_train_std,y_train_std,verbose=True)

     
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)
    
    
    y_train_pred = y_train_pred.flatten() * y_std + y_mean
    y_test_pred = y_test_pred.flatten() * y_std + y_mean
    
     
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    
    print(f"训练集 MSE: {train_mse:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}")
    print(f"训练集 R² : {train_r2:.4f}")
    print(f"测试集 R² : {test_r2:.4f}")
    
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=10, label='Prediction')
    
    p1 = max(y_test.max(), y_test_pred.max())
    p2 = min(y_test.min(), y_test_pred.min())
    plt.plot([p1, p2], [p1, p2], 'r--', lw=2, label='Perfect Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.title('Test Set: True vs Pred')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()