
# modelos
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


models_params = {
    "Random Forest": {
        "model": RandomForestRegressor(),
        "params": {
                # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,64,128,256]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(),
        "params": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(),
        "params": {
                # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
        }
    },
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },
    "XGBRegressor": {
        "model": XGBRegressor(),
        "params": {
            'learning_rate':[.1,.01,.05,.001],
            'n_estimators': [8,16,32,64,128,256]
        }
    },
    "AdaBoost Regressor": {
        "model": AdaBoostRegressor(),
        "params": {
            'learning_rate':[.1,.01,0.5,.001],
            # 'loss':['linear','square','exponential'],
            'n_estimators': [8,16,32,64,128,256]
        }
    },

}
