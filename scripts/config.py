class Config(object):

    def __init__(self, data_path="/labs/gevaertlab/data/momena/breast_data", patch_size=448, threshold=0.4,
                 selected_features=['out'], input_shape = 224, test_size = 0.02, val_size = 0.10, folder="folder",
                epochs = 20, gpu = "0", sampling_size_train  = 5, sampling_size_val = 5, batch_size = 32, lr = 5e-6,
                 lr_decay=1e-6, from_idx=0):
        
        self.data_path = data_path
        self.patch_size = patch_size
        self.threshold = threshold
        self.selected_features = selected_features
        self.input_shape = input_shape
        self.test_size = test_size
        self.val_size = val_size
        self.folder = folder
        self.epochs = epochs
        self.gpu = gpu
        
        self.sampling_size_train = sampling_size_train
        self.sampling_size_val = sampling_size_val 

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.from_idx = from_idx
