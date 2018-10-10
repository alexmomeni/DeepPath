import numpy as np
import keras
from utils.generator import Generator
from utils.custom_fit_generator import custom_fit_generator
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, precision_recall_curve, roc_curve
from keras.applications.densenet import DenseNet169
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class Model(object):

    def __init__(self, config):
        self.config = config
        self.data_init()
        self.model_init()

    #   os.makedirs("output/%s" % (self.config.folder))

    def data_init(self):
        pass

    def model_init(self):
        pass

    def train_predict(self):
        pass

    def get_metrics(self, y_scores, y_preds):
        list_of_metrics = ["accuracy", "precision", "recall", "f1score", "AUC", "AP"]
        y_true = self.y_test
        y_pred = y_preds
        y_score = y_scores
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')

        scores = [accuracy, precision, recall, f1score, auc, avg_precision]
        print(scores)


class ModelDL(Model):

    def __init__(self, config):
        Model.__init__(self, config)

    def data_init(self):

        print("\nData init")
        self.dataset = Dataset()
        generator = Generator(self.config, self.dataset)
        self.train_generator = generator.generate()
        self.X_val, self.y_val = self.dataset.convert_to_arrays(self.config,
                                                                self.dataset.get_partition(self.config)['val'])
        self.X_test, self.y_test = self.dataset.convert_to_arrays(self.config,
                                                                  self.dataset.get_partition(self.config)['test'])

    def model_init(self):

        print("\nModel init")
        self.base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling=None)
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        self.model = keras.models.Model(inputs=self.base_model.input, outputs=outputs)

    def set_trainable(self, from_idx=0):

        print("\nTraining")
        for layer in self.base_model.layers:
            layer.trainable = False
        for layer in self.model.layers[from_idx:]:
            layer.trainable = True

    def train(self, lr=1e-4, epochs=10, from_idx=0):

        self.set_trainable(from_idx=from_idx)
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=self.config.lr_decay)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        train_steps = len(self.dataset.get_partition(self.config)['train']) // self.config.batch_size
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        self.history = custom_fit_generator(model=self.model, generator=self.train_generator,
                                            steps_per_epoch=train_steps,
                                            epochs=epochs, verbose=1, validation_data=(self.X_val, self.y_val),
                                            shuffle=True,
                                            callbacks=[early_stopping], max_queue_size=30, workers=30,
                                            use_multiprocessing=True)

    def predict(self):

        print("\nPredicting")
        if len(self.y_test) == 1:
            y_scores = [self.model.predict(self.X_test, batch_size=self.config.batch_size)]
            y_preds = [(y_score > 0.5).astype(int) for y_score in y_scores]
        return y_scores, y_preds

    def train_predict(self):

        self.train(self.config.lr, self.config.epochs, self.config.from_idx)
        y_scores, y_preds = self.predict()
        np.save("output/%s/%s/y_scores" % (self.config.folder, self.config.experiment), y_scores)
        np.save("output/%s/%s/y_preds" % (self.config.folder, self.config.experiment), y_preds)

        return y_scores, y_preds