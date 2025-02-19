from keras import callbacks
from keras import metrics


class CustomEvalDataset(callbacks.Callback):
    def __init__(self, x, y, metrics=metrics.SparseCategoricalAccuracy()):
        super().__init__()
        self.x = x
        self.y_true = y
        self.metrics = metrics
        self.list_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        self.metrics.reset_state()
        y_pred = self.model.predict(self.x, verbose=0)
        self.metrics.update_state(self.y_true, y_pred)
        result = float(self.metrics.result())
        self.list_metrics.append(result)
