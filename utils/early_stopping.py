import torch

class EarlyStoppingForOptimization:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation accuracy improved.
            verbose (bool): If True, prints a message for each validation accuracy improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc, model):
        """
        Args:
            val_acc (float): Validation accuracy for the current epoch.
            model (nn.Module): The model being trained.
        Returns:
            bool: True if training should stop, False otherwise.
        """
        score = val_acc
        if self.best_score is None:
            # First evaluation, set the best score
            self.best_score = score
            self.save_checkpoint(model, score)
        elif score < self.best_score + self.delta:
            # Validation accuracy did not improve
            self.counter += 1
            if self.counter >= self.patience:
                # Stop training if patience is exhausted
                self.early_stop = True
        else:
            # Validation accuracy improved
            self.best_score = score
            self.save_checkpoint(model, score)
            self.counter = 0  # Reset the counter
        return self.early_stop

    def save_checkpoint(self, model, score):
        """
        Saves the model when validation accuracy improves.
        Args:
            model (nn.Module): The model to save.
            score (float): The current validation accuracy.
        """
        if self.verbose:
            print(f'Validation accuracy increased ({self.best_score:.6f} --> {score:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time the validation metric improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation metric improvement.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0