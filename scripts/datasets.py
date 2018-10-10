import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
from PIL import Image


class Dataset(object):

    def list_features(self, config):
        self.config = config
        data = pd.read_table('/home/aamomeni/research/brca_signatures/Data/TCGA_BRCA_out1.txt', header=0, index_col=0)
        list_features = data.columns
        return list_features

    def get_output_data(self, config):
        self.config = config
        selected_features = pd.read_table('/home/aamomeni/research/brca_signatures/Data/TCGA_BRCA_out1.txt', header=0,
                                          index_col=0)
        l = [i[:16] for i in selected_features.index]
        selected_features.index = l
        selected_features = selected_features[self.config.selected_features]
        selected_features = selected_features.dropna()
        return selected_features

    def get_binarized_data(self, config):

        self.config = config
        output_data = self.get_output_data(self.config)
        binarized_data = output_data.apply(lambda x: LabelBinarizer().fit_transform(x)[:, 0], axis=0)
        return binarized_data

    def get_partition(self, config):

        self.config = config
        labels = self.get_labels(config)
        for k, v in labels.items():
            t = v
        samples = []

        q = os.listdir('/labs/gevaertlab/data/momena/breast_data/patches_448/')
        l = [i[:16] for i in q]

        for i in list(t.keys()):
            if np.intersect1d(l, i) != []:
                samples.append(q[l.index(i)])

        np.random.shuffle(samples)
        idx_val = int((1 - config.val_size - config.test_size) * len(samples))
        idx_test = int((1 - config.test_size) * len(samples))
        train_samples, val_samples, test_samples = np.split(samples, [idx_val, idx_test])
        train_samples, val_samples, test_samples = list(train_samples), list(val_samples), list(test_samples)
        partition = {'train': train_samples, 'val': val_samples, 'test': test_samples}

        return partition

    def get_ids(self, config):

        self.config = config
        patient_ids = os.listdir("%s/patches_%d" % (self.config.data_path, self.config.patch_size))
        patient_ids = [patient_id[:16] for patient_id in patient_ids]
        ids = list(patient_ids)
        return ids

    def get_labels(self, config):
        self.config = config
        labels = {}
        samples = self.get_ids(config)
        data = self.get_binarized_data(config)

        for feature in self.config.selected_features:
            labels[feature] = {}
            for sample in samples:
                try:
                    labels[feature][sample] = data.loc[sample, feature]
                except:
                    pass

        return labels

    def convert_to_arrays(self, config, samples):

        labels = self.get_labels(config)

        X, ids = [], []
        for sample in samples:
            patches = os.listdir("%s/patches_%d/%s" % (self.config.data_path, self.config.patch_size, sample))
            patches = np.random.choice(patches, size=5, replace=True)
            for patch in patches:
                ID = "%s/patches_%d/%s/%s" % (self.config.data_path, self.config.patch_size, sample, patch)
                ids.append(ID)
                img = Image.open(ID)
                img = img.resize((self.config.input_shape, self.config.input_shape))
                image = np.array(img)[:, :, :3]
                X.append(image)
        X = np.asarray(X)

        y = []

        for label in labels.keys():
            y_label = []
            for ID in ids:
                sample = ID.split('/')[-2][0:16]
                y_label.append(labels[label][sample])
            y_label = np.asarray(y_label)
            y.append(y_label)
        return X, y
