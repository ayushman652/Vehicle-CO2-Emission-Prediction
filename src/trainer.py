from sklearn.linear_model import LinearRegression

def train_model(x_train, y_train):
    """train a multiple linear regression model"""
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

