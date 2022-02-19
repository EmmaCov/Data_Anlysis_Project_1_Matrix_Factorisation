"""
DEEP MATRIX FACTORIZATION
Original paper: https://www.ijcai.org/proceedings/2017/0447.pdf
Code partially inspired by:  https://github.com/hegongshan/deep_matrix_factorization
"""

from keras import backend
from tensorflow.keras import optimizers
from keras.layers import Input, Dense, Lambda, Flatten
from keras.models import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Data:
    """
    An object to encapsulate data and useful infos about it: dimensions, train/test.
    """

    def __init__(self, csv_path):
        head = ['user id', 'item id', 'rating', 'timestamp']
        df_movies = pd.read_csv(csv_path, sep='\t', names=head)
        self.train_set, self.test_set = train_test_split(df_movies, test_size=0.1)
        self.num_users = max(df_movies['user id'])
        self.num_items = max(df_movies['item id'])
        self.R_matrix = self.create_R_matrix()

    def create_R_matrix(self):
        """
        Turns the movie dataframe into a U*I matrix (filled with 0 when no rating)
        """
        R = pd.DataFrame(index=np.arange(1, self.num_items + 1), columns=np.arange(1, self.num_users + 1))
        groups = self.train_set.groupby('user id')
        for name, group in groups:
            movies_rated = group['item id'].unique()
            R.loc[movies_rated, name] = group['rating'].values
        R = R.fillna(0)
        return R.values.transpose()


class Context:
    """
    Defines the parameter values of a particular execution (fit + test).
    """

    def __init__(self, epochs, user_layers, item_layers, batch_size, learning_rate, negative_ratio, name):
        self.epochs = epochs
        self.user_layers = user_layers
        self.item_layers = item_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.name = name


class DMF:
    """
    Class to encapsulate the model
    """

    def __init__(self, data: Data, context: Context):
        self.data = data
        self.context = context
        # input matrix (and its transposition)
        self.user_rating = backend.constant(data.R_matrix)
        self.item_rating = backend.constant(data.R_matrix.T)
        # create model
        self.model = self.create_model()

    @staticmethod
    def init_normal(shape, dtype=None):
        return backend.random_normal(shape=shape, stddev=0.01, dtype=dtype)

    @staticmethod
    def cosine_similarity(inputs, epsilon=1.0e-6, delta=1e-12):
        x, y = inputs[0], inputs[1]
        numerator = backend.sum(x * y, axis=1, keepdims=True)
        denominator = backend.sqrt(
            backend.sum(x * x, axis=1, keepdims=True) * backend.sum(y * y, axis=1, keepdims=True))
        cosine_similarity = numerator / backend.maximum(denominator, delta)
        return backend.maximum(cosine_similarity, epsilon)

    def create_model(self):
        """Create and compile a model"""

        # Model is too complex for Keras Sequential API, we use Functional API.

        # inputs are just 1 rating
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        # we turn it into a vector by using a lambda that picks each user and each rating in the matrix
        user_rating_input = Lambda(lambda x: backend.gather(self.user_rating, x))(user_input)
        user_rating_vector = Flatten()(user_rating_input)
        item_rating_input = Lambda(lambda x: backend.gather(self.item_rating, x))(item_input)
        item_rating_vector = Flatten()(item_rating_input)

        user_vector = None
        item_vector = None
        for i in range(len(self.context.user_layers)):
            layer = Dense(self.context.user_layers[i],
                          activation='relu',
                          kernel_initializer=self.init_normal,
                          bias_initializer=self.init_normal,
                          name='user_layer%d' % (i + 1))
            if i == 0:
                user_vector = layer(user_rating_vector)
            else:
                user_vector = layer(user_vector)

        for i in range(len(self.context.item_layers)):
            layer = Dense(self.context.item_layers[i],
                          activation='relu',
                          kernel_initializer=self.init_normal,
                          bias_initializer=self.init_normal,
                          name='item_layer%d' % (i + 1))
            if i == 0:
                item_vector = layer(item_rating_vector)
            else:
                item_vector = layer(item_vector)

        y_predict = Lambda(function=self.cosine_similarity, name='predict')([user_vector, item_vector])
        model = Model(inputs=[user_input, item_input], outputs=y_predict)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.context.learning_rate), loss='binary_crossentropy')
        return model

    def evaluate_rmse(self, test_set):
        """
        Evaluate RMSE on the model base on a test set
        """
        users, items, ratings = test_set['user id'], test_set['item id'], test_set['rating']
        predicted_ratings = self.model.predict([np.array(users) - 1, np.array(items) - 1], batch_size=10000, verbose=0)
        return mean_squared_error(np.array(ratings), predicted_ratings * max(ratings))

    def create_training_data(self):
        """
        Format data to a list of unitary ratings ready for trainings, with negative sample.
        """
        train_users = []
        train_items = []
        train_ratings = []
        for line in self.data.train_set.values:
            # add each line of train set to training data
            user, item, rating = line[0] - 1, line[1] - 1, line[2]
            train_users.append(user)
            train_items.append(item)
            train_ratings.append(rating)

            # for each positive rating, we want several null ratings
            negative_items = []
            while len(negative_items) < self.context.negative_ratio:
                i = np.random.randint(self.data.num_items)
                if self.data.R_matrix[user, i] == 0 and i not in negative_items:
                    # we add the user, the unrated item, and a zero to the training data
                    train_users.append(user)
                    train_items.append(i)
                    train_ratings.append(0)

                    # we make sure to not add it twice by storing it
                    negative_items.append(i)

        return np.array(train_users), np.array(train_items), np.array(train_ratings) / max(train_ratings)


def train_test_save(dmf):
    """
    Kind of "main" execution, useful to change params, then relaunch it.
    Train the model, test it at each epoch, save results.
    """
    print(dmf.context.name)

    train_users, train_items, train_ratings = dmf.create_training_data()
    rmse = [dmf.evaluate_rmse(dmf.data.test_set)]
    for e in range(dmf.context.epochs):
        dmf.model.fit(x=[np.array(train_users), np.array(train_items)],
                      y=np.array(train_ratings),
                      batch_size=dmf.context.batch_size,
                      epochs=1)
        rmse.append(dmf.evaluate_rmse(dmf.data.test_set))
    filename = dmf.context.name + ".txt"
    with open(filename, 'w') as f:
        f.write(str(rmse))


if __name__ == '__main__':
    # data = Data('ml-100k/u.data')
    data = Data('ml-1m/ratings.dat')
    # data = Data('ml-25m/ratings.csv')
    context = Context(epochs=5,
                      user_layers=[512, 64],
                      item_layers=[1024, 64],
                      batch_size=256,
                      learning_rate=0.001,
                      negative_ratio=0,
                      name="basic")

    # first execution with basic parameters
    dmf = DMF(data, context)
    train_test_save(dmf)

    # faster learning
    context.learning_rate = 0.01
    context.name = "learning_rate_0_01"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.learning_rate = 0.1
    context.name = "learning_rate_0_1"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.learning_rate = 0.001

    # more or less layers
    context.user_layers = [512, 256, 64]
    context.item_layers = [1024, 256, 64]
    context.name = "layers_3"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.user_layers = [512, 256, 128, 64]
    context.item_layers = [1024, 256, 128, 64]
    context.name = "layers-4"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.user_layers = [64]
    context.item_layers = [64]
    context.name = "layers_1"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.user_layers = [512, 64]
    context.item_layers = [1024, 64]

    # features in latent space
    context.user_layers = [512, 32]
    context.item_layers = [1024, 32]
    context.name = "features_32"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.user_layers = [512, 128]
    context.item_layers = [1024, 128]
    context.name = "features_128"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.user_layers = [512, 256]
    context.item_layers = [1024, 256]
    context.name = "features_256"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.user_layers = [512, 64]
    context.item_layers = [1024, 64]

    # batch size
    context.batch_size = 1
    context.name = "batch_1"
    dmf = DMF(data, context)
    train_test_save(dmf)

    context.batch_size = 100
    context.name = "batch_100"
    dmf = DMF(data, context)
    train_test_save(dmf)
