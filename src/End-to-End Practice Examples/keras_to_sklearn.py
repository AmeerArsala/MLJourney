import scikeras
from scikeras.wrappers import KerasRegressor, KerasClassifier

class ModelParams:
    def __init__(self, model, batch_size, epochs, callbacks=[]):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
    

class ModelWrapper:
    def __init__(self, scikeras_model):
        super().__init__()
        self.model = scikeras_model
        
    def prefix_keys(prefix, dictionary):
        return {prefix + key: value for key, value in dictionary.items()}
    
    def fit(self, X, y, **params):
        prefixed_params = prefix_keys("fit__", params)
        self.model.set_params(**prefixed_params)
        
        self.model.fit(X, y)
        return self
    
    def partial_fit(self, X, y, sample_weight=None, **params):
        prefixed_params = prefix_keys("partial_fit__", params)
        self.model.set_params(**prefixed_params)
        
        self.model.partial_fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    
# Input: ModelParams model_params
# Output: ModelWrapper
def keras_classifier(model_params):
    classifier = KerasClassifier(
        model=model_params.model,
        batch_size=model_params.batch_size,
        epochs=model_params.epochs,
        callbacks=model_params.callbacks
    )
    
    return ModelWrapper(classifier)

# Input: ModelParams model_params
# Output: ModelWrapper
def keras_regressor(model_paraams):
    regressor = KerasRegressor(
        model=model_params.model,
        batch_size=model_params.batch_size,
        epochs=model_params.epochs,
        callbacks=model_params.callbacks
    )
    
    return ModelWrapper(regressor)
