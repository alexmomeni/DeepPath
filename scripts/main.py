import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from config import Config
from models import ModelDL

config = Config(folder="model_comparison", epochs=20, gpu = "1")
session_config = tf.ConfigProto()
session_config.gpu_options.visible_device_list = config.gpu
session_config.gpu_options.allow_growth = True
set_session(tf.Session(config= session_config))

variable_name = "classifier"
values = ["DL"]
experiments = ["DNN"]

metrics_summary = []
model = ModelDL(config)
y_scores, y_preds = model.train_predict()
model.get_metrics(y_scores, y_preds)