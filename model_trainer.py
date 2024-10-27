import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'KNN': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Linear Regression': LinearRegression()
        }
        self.best_model = None
        self.best_score = float('inf')

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)

            print(f"{name} MSE: {mse}")

            if mse < self.best_score:
                self.best_score = mse
                self.best_model = model

        self.save_model()

    def save_model(self):
        joblib.dump(self.best_model, 'artifacts/model.pkl')
        print("Best model saved successfully.")
