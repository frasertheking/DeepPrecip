import sys,os,io
import time
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
import tensorflow.keras as keras
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/"
MIN_AVG = 20
USE_SHUFFLE = False
RANDOM_STATE = None
NUM_KF_SPLITS = 5
BATCH_SIZE = 128
EPOCHS = 3000
PRED_COUNT = 256
RUN_TIME = str(int(time.time()))
os.mkdir('runs/' + RUN_TIME)


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
        
    def mse(self):
        return metrics.mean_squared_error(self.y_test, self.y_pred)
    
    def corr(self):
        return np.corrcoef(self.y_test, self.y_pred)[0][1]
    
    def mae(self):
        return metrics.mean_absolute_error(self.y_test, self.y_pred)
    
    def r2(self):
        return metrics.r2_score(self.y_test, self.y_pred)
    
    def export_metrics():
        return self.mse(), self.corr(), self.mae(), self.r2()
    
    def data_length(self):
        return len(self.y_pred)
        
    def max_val(self):
        max_val = max(self.y_pred)
        if (max(self.y_test)) > max_val:
            max_val = max(self.y_test)
        return max_val
    
    def summary(self):
        print("\n####################\n")
        print(self.name + " STATS (n=" + str(self.data_length()) + "):" + "\nMSE: " + str(round(self.mse(),5)) + \
              "\nCorrelation: " + str(round(self.corr(),5)) +\
              "\nMean Absolute Error: " + str(round(self.mae(),5)) +\
              "\nR-Squared: " + str(round(self.r2(),5)))
        print("\n####################\n")
        
    def scatter(self):
        stats = self.name + " STATS (n=" + str(self.data_length()) + "):" + "\nMSE: " + str(round(self.mse(),5)) + \
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
        plt.savefig('runs/' + RUN_TIME + '/scatter_full_column.png', DPI=300)
        
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
        plt.savefig('runs/' + RUN_TIME + '/timeseries_full_column.png', DPI=300)
        
    def freq(self):
        plt.figure(figsize=(15, 18))
        sb.distplot(self.y_pred, hist = False, color = 'r', label = 'Predicted Values')
        sb.distplot(self.y_test, hist = False, color = 'b', label = 'Actual Values')
        plt.title(self.name + ' Accumulation Distribution')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend(loc = 'upper right')
        plt.savefig('runs/' + RUN_TIME + '/freq_full_column.png', DPI=300)

def make_divisible(number, divisor):
    return number - number % divisor

def plot_accuracies(histories):
    mse = []
    val_mse = []
    history_mse = -1
    history_val_mse = -1
    
    if isinstance(histories, list):
        for history in histories:
            mse.append(history.history['mean_squared_error'])
            val_mse.append(history.history['val_mean_squared_error'])

        flat_mse = [item for sublist in mse for item in sublist]
        flat_val_mse = [item for sublist in val_mse for item in sublist]
        history_mse = flat_mse
        history_val_mse = flat_val_mse
    else:
        history_mse = histories.history['mean_squared_error']
        history_val_mse = histories.history['val_mean_squared_error']

    plt.figure(figsize=(15, 18))
    plt.grid(linestyle='--')
    plt.ylim((0, 0.001))
    plt.plot(history_mse, color='black', label='train')
    plt.plot(history_val_mse, color='red', label='test')
    plt.title('model accuracy')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('runs/' + RUN_TIME + '/curves_full_column.png', DPI=300)


####################################################################################################################################
############ Import Data From CSV

site_df_array = []
site_name_array = []

X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
train_len_arr = []
test_len_arr = []

for filename in sorted(os.listdir(DATA_PATH)):
    if filename.endswith(".csv"):
        print("Opening site", filename)
        df = pd.read_csv(DATA_PATH + "/" + filename, index_col=[0])
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df[df['in_situ_precip'] > 0]
        df = df[df['in_situ_precip'] < 0.15]
        df['wind_speed'] = df['wind_speed'].astype('float64')
        df['2mt'] = df['2mt'].astype('float64')
        df = df[df['wind_speed'] < 5]
        df = df[df['mean_bins'].notnull()]
        df = df.loc[df['mean_bins'].shift(-1) != df['mean_bins']]
        df = df.loc[df['2mt'].shift(-20) != df['2mt']]
        df = df.rolling(window=MIN_AVG, on='timestamp').mean()
        df = df.dropna()
        site_df_array.append(df)
        print(df.shape[0])
        X_var = df.drop(columns=['mean_bins', 'in_situ_precip'], axis=1)
        y_var = df['in_situ_precip']
        X_train_loc, X_test_loc, y_train_loc, y_test_loc = train_test_split(X_var, y_var, test_size=0.1, shuffle=USE_SHUFFLE, random_state=RANDOM_STATE)

        if X_train.empty:
            X_train = X_train_loc
            X_test = X_test_loc
            y_train = y_train_loc
            y_test = y_test_loc
        else:
            X_train = pd.concat([X_train, X_train_loc])
            X_test = pd.concat([X_test, X_test_loc])
            y_train = pd.concat([y_train, y_train_loc])
            y_test = pd.concat([y_test, y_test_loc])

        train_len_arr.append(len(X_train))
        test_len_arr.append(len(X_test))
        site_name_array.append(os.path.splitext(filename)[0])
        
X_train.to_csv('model_out/X_train_full_column.csv')
X_test.to_csv('model_out/X_test_full_column.csv')
y_train.to_csv('model_out/y_train_full_column.csv')
y_test.to_csv('model_out/y_test_full_column.csv')

X_train.drop(columns=['timestamp', 'lon', 'lat', 'wind_speed', '2mt', 'bin_1', 'bin_2', 'dopp_1', 'dopp_2', 'spec_1', 'spec_2'], inplace=True, axis=1)
X_test.drop(columns=['timestamp', 'lon', 'lat', 'wind_speed', '2mt', 'bin_1', 'bin_2', 'dopp_1', 'dopp_2', 'spec_1', 'spec_2'], inplace=True, axis=1)

####################################################################################################################################
############ Train Testing

features = X_train.columns
history = -1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

train_data_len = X_train.shape[0]
train_steps_per_execution = train_data_len // BATCH_SIZE
train_data_len = make_divisible(train_data_len, BATCH_SIZE)
X_train, y_train = X_train[:train_data_len], y_train[:train_data_len]

test_data_len = X_test.shape[0]
test_steps_per_execution = test_data_len // BATCH_SIZE
test_data_len = make_divisible(test_data_len, BATCH_SIZE)
X_test, y_test = X_test[:test_data_len], y_test[:test_data_len]


callback = keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=8)
model = keras.Sequential([
    keras.layers.Conv1D(filters=256, kernel_size=16, activation='relu', input_shape=(X_train.shape[1], 1)),
    keras.layers.Conv1D(filters=256, kernel_size=16, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(PRED_COUNT, kernel_constraint=keras.constraints.unit_norm(), activation='relu', kernel_regularizer=keras.regularizers.l2(0.5)),
    keras.layers.Dense(PRED_COUNT, kernel_constraint=keras.constraints.unit_norm(), activation='relu', kernel_regularizer=keras.regularizers.l2(0.5)),
    keras.layers.Dense(PRED_COUNT, kernel_constraint=keras.constraints.unit_norm(), activation='relu'),
    keras.layers.Dense(1)])

opt = keras.optimizers.Adam(learning_rate=0.0000001)
model.compile(optimizer=opt, loss=keras.losses.MeanSquaredError(), metrics=['mean_squared_error'])
history = model.fit(x=X_train, y=y_train, verbose=1, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[callback])
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
model.save('models')

df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred.flatten()})
df.to_csv('model_out/mlp_full_column.csv', index=False)

stats_mlp = model_metrics('mlp', np.arange(len(y_test)), y_test, y_pred.flatten())
stats_mlp.summary()
stats_mlp.timeseries()
stats_mlp.scatter()
stats_mlp.freq()
# plot_accuracies(history)


