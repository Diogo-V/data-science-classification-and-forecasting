from torch import zeros, manual_seed, Tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.autograd import Variable

from pandas import read_csv, DataFrame
from torch import manual_seed, Tensor
from torch.autograd import Variable
from ts_functions import PREDICTION_MEASURES, split_dataframe, sliding_window, plot_forecasting_series, plot_evaluation_results
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score

from ds_charts import HEIGHT, multiple_line_chart
from matplotlib.pyplot import subplots, show, savefig
import matplotlib.pyplot as plt

target = 'Glucose'
index_col='timestamp'

class LSTMForecaster():
    def __init__(self, data: DataFrame) -> None:

        manual_seed(1)
        self.nr_features = len(data.columns)
        sc = MinMaxScaler()
        data = DataFrame(sc.fit_transform(data), index=data.index, columns=data.columns)
        self.train_data, self.test_data = split_dataframe(data, trn_pct=0.75)

        self.seq_length = 4
        self.num_epochs = 2000
        self.learning_rate = 0.001
        self.sequence_size = [4, 20, 60, 100]
        self.nr_hidden_units = [8, 16, 32]
        self.max_iter = [500, 500, 1500, 2500]

        """ self.train_data = data_train
        self.train_y = self.train_data.pop('readmitted').values
        self.test_data = data_test
        self.test_y = self.test_data.pop('readmitted').values
        self.n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
        self.max_depths = [5, 10, 25]
        self.learning_rate = [.1, .5, .9] """

    def explore_best_lstm(self):
        trnX, trnY = sliding_window(self.train_data, seq_length = self.seq_length)
        trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
        tstX, tstY = sliding_window(self.test_data, seq_length = self.seq_length)
        tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))

        my_lstm = DS_LSTM(input_size=1, hidden_size=8, learning_rate=0.001)

        for epoch in range(self.num_epochs+1):
            loss = my_lstm.fit(trnX, trnY)
            if epoch % 500 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss))

        prd_trn = my_lstm(trnX)
        prd_tst = my_lstm(tstX)

        print('TRAIN R2=', r2_score(trnY.data.numpy(), prd_trn.data.numpy()))
        print('TEST R2=', r2_score(tstY.data.numpy(), prd_tst.data.numpy()))

        best = ('',  0, 0.0)
        last_best = -100
        best_model = None
        measure = 'R2'
        flag_pct = False

        episode_values = [self.max_iter[0]]
        for el in self.max_iter[1:]:
            episode_values.append(episode_values[-1]+el)

        nCols = len(self.sequence_size)
        _, axs = subplots(1, nCols, figsize=(nCols*HEIGHT, HEIGHT), squeeze=False)
        values = {}
        for s in range(len(self.sequence_size)):
            length = self.sequence_size[s]
            trnX, trnY = sliding_window(self.train_data, seq_length = length)
            trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
            tstX, tstY = sliding_window(self.test_data, seq_length = length)
            tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))

            for k in range(len(self.nr_hidden_units)):
                hidden_units = self.nr_hidden_units[k]
                yvalues = []
                model = DS_LSTM(input_size=self.nr_features, hidden_size=hidden_units, learning_rate=self.learning_rate)
                next_episode_i = 0
                for n in range(1, episode_values[-1]+1):
                    model.fit(trnX, trnY)
                    if n == episode_values[next_episode_i]:
                        next_episode_i += 1
                        prd_tst = model.predict(tstX)
                        yvalues.append((PREDICTION_MEASURES[measure])(tstY, prd_tst))
                        print((f'LSTM - seq length={length} hidden_units={hidden_units} and nr_episodes={n}->{yvalues[-1]:.2f}'))
                        if yvalues[-1] > last_best:
                            best = (length, hidden_units, n)
                            last_best = yvalues[-1]
                            best_model = model
                values[hidden_units] = yvalues

            multiple_line_chart(
                episode_values, values, ax=axs[0, s], title=f'LSTM seq length={length}', xlabel='nr episodes', ylabel=measure, percentage=flag_pct)
        savefig(f'health-forecasting/records/forecasting/lstm/lstm_study.png')
        print(f'Best results with seq length={best[0]} hidden={best[1]} episodes={best[2]} ==> measure={last_best:.2f}')
        f= open('health-forecasting/records/forecasting/lstm/lstm_best_details.txt', 'w')
        f.write(f'Best approach: Seq length = {best[0]} with hidden units = {best[1]} and epochs = {best[2]}')
        f.close()

        return best[0], best[1], best[2], best_model

    def compute_best_lstm(self, sequence_length, hidden_units, epochs, best_model):

        trnX, trnY = sliding_window(self.train_data, seq_length = sequence_length)
        trainY = DataFrame(trnY)
        trainY.index = self.train_data.index[sequence_length+1:]
        trainY.columns = [target]
        trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
        prd_trn = best_model.predict(trnX)
        prd_trn = DataFrame(prd_trn)
        prd_trn.index = self.train_data.index[sequence_length+1:]
        prd_trn.columns = [target]

        tstX, tstY = sliding_window(self.test_data, seq_length = sequence_length)
        testY = DataFrame(tstY)
        testY.index = self.test_data.index[sequence_length+1:]
        testY.columns = [target]
        tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))
        prd_tst = best_model.predict(tstX)
        prd_tst = DataFrame(prd_tst)
        prd_tst.index = self.test_data.index[sequence_length+1:]
        prd_tst.columns = [target]

        plot_evaluation_results(trnY.data.numpy(), prd_trn, tstY.data.numpy(), prd_tst, f'health-forecasting/records/forecasting/lstm/lstm_eval.png')
        savefig('health-forecasting/records/forecasting/lstm/lstm_eval.png')
        self.plot_forecasting_series(trainY, testY, prd_trn.values, prd_tst.values, f'health-forecasting/records/forecasting/lstm/lstm_plots.png', x_label=index_col, y_label=target)
        savefig('health-forecasting/records/forecasting/lstm/lstm_plots.png')

    def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, figname: str, x_label: str = 'time', y_label:str =''):
        _, ax = plt.subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(figname)
        ax.plot(trn.index, trn, label='train', color='b')
        ax.plot(trn.index, prd_trn, '--y', label='train prediction')
        ax.plot(tst.index, tst, label='test', color='g')
        ax.plot(tst.index, prd_tst, '--r', label='test prediction')
        ax.legend(prop={'size': 5})


class DS_LSTM(Module):

    def __init__(self, input_size, hidden_size, learning_rate, num_layers=1, num_classes=1):
        super(DS_LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = Linear(hidden_size, self.num_classes)
        self.criterion = MSELoss()    # mean-squared error for regression
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        h_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

    def fit(self, trainX, trainY):
        # Train the model
        outputs = self(trainX)
        self.optimizer.zero_grad()
        # obtain the loss function
        loss = self.criterion(outputs, trainY)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, data):
        # Predict the target variable for the input data
        return self(data).detach().numpy()