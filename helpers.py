####################################################################################################################################
############ Class Definitions

# Used for holding ERA5-L data
class model_metrics(object):
    name = ""
    index_val = []
    y_test = []
    y_pred = []
    
    def __init__(self, name, index_val, y_test, y_pred):
        self.name = name
        self.index_val = index_val
        self.y_test = y_test
        self.y_pred = y_pred
        
    def rmse(self):
        return np.sqrt(metrics.mean_squared_error(self.y_test, self.y_pred))
    
    def corr(self):
        return np.corrcoef(self.y_test, self.y_pred)[0][1]
    
    def mae(self):
        return metrics.mean_absolute_error(self.y_test, self.y_pred)
    
    def r2(self):
        return metrics.r2_score(self.y_test, self.y_pred)
    
    def export_metrics():
        return self.rmse(), self.corr(), self.mae(), self.r2()
    
    def data_length(self):
        return len(self.y_pred)
        
    def max_val(self):
        max_val = max(self.y_pred)
        if (max(self.y_test)) > max_val:
            max_val = max(self.y_test)
        return max_val
    
    def summary(self):
        print("\n####################\n")
        print(self.name + " STATS (n=" + str(self.data_length()) + "):" + "\nRMSE: " + str(round(self.rmse(),5)) + \
              "\nCorrelation: " + str(round(self.corr(),5)) +\
              "\nMean Absolute Error: " + str(round(self.mae(),5)) +\
              "\nR-Squared: " + str(round(self.r2(),5)))
        print("\n####################\n")
        
    def scatter(self):
        stats = self.name + " STATS (n=" + str(self.data_length()) + "):" + "\nRMSE: " + str(round(self.rmse(),5)) + \
              "\nCorrelation: " + str(round(self.corr(),5)) +\
              "\nMean Absolute Error: " + str(round(self.mae(),5)) +\
              "\nR-Squared: " + str(round(self.r2(),5))
        fig, ax=plt.subplots(figsize=(10,10))
        plt.grid(linestyle='--')
        plt.title(self.name + ' Actual vs Predicted Values')
        plt.xlabel('Predicted Accumulation (mm SWE)')
        plt.ylabel('Observed Accumulation (mm SWE)')
        plt.xlim((0, self.max_val()))
        plt.ylim((0, self.max_val()))
        plt.scatter(self.y_pred, self.y_test,color='red', alpha=0.25)
        plt.plot([0, self.max_val()], [0, self.max_val()], linestyle='--', color='black')
        plt.text(0.02, 0.9, stats, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=16)
        plt.show()
        
    def timeseries(self):
        roll_y_test = pd.Series(self.y_test).rolling(250).mean().tolist()
        roll_y_pred = pd.Series(self.y_pred).rolling(250).mean().tolist()
        fig, ax=plt.subplots(figsize=(20,7))
        plt.grid(linestyle='--')
        plt.title(self.name + ' Timeseries')
        plt.xlabel('Time')
        plt.ylabel('Accumulation (mm SWE)')
        plt.plot(np.arange(len(roll_y_test)), roll_y_test, color='black', label='observed')
        plt.plot(np.arange(len(roll_y_pred)), roll_y_pred, color='red', label='predicted')
        plt.axhline(np.nanmean(self.y_test), color='black', linestyle='--')
        plt.axhline(np.nanmean(self.y_pred), color='red', linestyle='--')
        plt.legend()
        plt.show()
        
    def freq(self):
        plt.figure(figsize=(15, 18))
        sb.distplot(self.y_pred, hist = False, color = 'r', label = 'Predicted Values')
        sb.distplot(self.y_test, hist = False, color = 'b', label = 'Actual Values')
        plt.title(self.name + ' Accumulation Distribution')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend(loc = 'upper right')
        plt.show()


####################################################################################################################################
############ Helper Functions

def plot_predictor_importances(model, model_df):
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.2, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    plt.figure(figsize=(15, 45))
    plt.grid(linestyle='--')
    plt.title('RF Feature Importances')
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(columns[sorted_idx], model.feature_importances_[sorted_idx], color='red')
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel('Predictor')
    plt.show()
    
    
def plot_accuracies(histories):
    mse = []
    val_mse = []
    history_mse = -1
    history_val_mse = -1
    
    if isinstance(histories, list):
        for history in histories:
            mse.append(history.history['mean_squared_error'])
            val_mse.append(history.history['val_mean_squared_error'])

        history_mse = np.mean(mse, axis=0)
        history_val_mse = np.mean(val_mse, axis=0)
    else:
        history_mse = histories.history['mean_squared_error']
        history_val_mse = histories.history['val_mean_squared_error']
    
    plt.figure(figsize=(15, 18))
    plt.grid(linestyle='--')
#     plt.ylim((0, 0.008))
    plt.plot(history_mse, color='black', label='train')
    plt.plot(history_val_mse, color='red', label='test')
    plt.title('model accuracy')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    
def perform_rf_hyperparameterization(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.2, random_state=RANDOM_STATE)
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 5, verbose=2, random_state=RANDOM_STATE, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    print("Best params:")
    print(rf_random.best_params_)
    
def run_cv(model, model_name, fold, X, Y, use_cv, epochs, batch_size):    
    if use_cv:
        fold_num = 1
        y_tests = []
        y_preds = []
        fitted_models = []
        histories = []
        for train_index, test_index in fold.split(X, Y):
            print("On Fold", fold_num)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            if model_name == 'lstm':
                X_train = np.asarray(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
                X_test  = np.asarray(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
                histories.append(model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0, shuffle=USE_SHUFFLE))
            elif model_name == 'mlp':
                tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
                histories.append(model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), verbose=0, batch_size=epochs, epochs=batch_size, callbacks=[tensorboard_callback]))
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_preds.append(y_pred)
            y_tests.append(y_test)
            fitted_models.append(model)
            fold_num += 1

        # Save a model
    #     filename = 'rf.sav'
    #     pickle.dump(fitted_models[0], open(filename, 'wb'))

        flat_preds = [item for sublist in y_preds for item in sublist]
        flat_tests = [item for sublist in y_tests for item in sublist]
        if model_name == 'mlp' or model_name == 'lstm':
            flat_preds = np.concatenate(flat_preds).ravel()
        return fitted_models, model_metrics(model_name, np.arange(len(flat_tests)), flat_tests, flat_preds), histories
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        history = -1
                      
        if model_name == 'lstm':
            X_train = np.asarray(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test  = np.asarray(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0, shuffle=USE_SHUFFLE)
        elif model_name == 'mlp':
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
            history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), verbose=0, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        flat_preds = y_pred
        if model_name == 'lstm' or model_name == 'mlp':
            flat_preds = [item for sublist in y_pred for item in sublist]
#         if model_name == 'lstm':
#             flat_preds = np.concatenate(flat_preds).ravel()
            
    return model, model_metrics(model_name, np.arange(len(y_test)), y_test, flat_preds), history
   
def site_based_test(model, model_name, fold, X, Y, custom_x_test, custom_y_test, use_cv, epochs, batch_size):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
        
        X_test = custom_x_test 
        y_test = custom_y_test 
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        history = -1
                      
        if model_name == 'lstm':
            X_train = np.asarray(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test  = np.asarray(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0, shuffle=USE_SHUFFLE)
        elif model_name == 'mlp':
            history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), verbose=0, batch_size=batch_size, epochs=epochs)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        flat_preds = y_pred
        if model_name == 'lstm' or model_name == 'mlp':
            flat_preds = [item for sublist in y_pred for item in sublist]
        return model, model_metrics(model_name, np.arange(len(y_test)), y_test, flat_preds), history
    
def create_mlp(optimizer='adam', activation='relu', neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
    return model