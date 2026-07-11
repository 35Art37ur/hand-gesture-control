import numpy as np
from tensorflow.keras.utils import Sequence

from augment_dataset import augment_sequence


class AugmentedSequence(Sequence):

    def __init__(
        self,
        X,
        y,
        batch_size=16,
        augment=True,
        shuffle=True
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

        self.indices = np.arange(len(X))
        self.on_epoch_end()


    def __len__(self):
        return len(self.X) // self.batch_size


    def __getitem__(self, index):

        batch_indices = self.indices[
            index*self.batch_size :
            (index+1)*self.batch_size
        ]

        X_batch = []
        y_batch = []

        for i in batch_indices:

            sample = self.X[i]

            if self.augment:
                sample = augment_sequence(sample)[0]

            X_batch.append(sample)
            y_batch.append(self.y[i])

        return (
            np.array(X_batch, dtype=np.float32),
            np.array(y_batch, dtype=np.float32)
        )


    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)