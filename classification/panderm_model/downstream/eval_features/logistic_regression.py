"""
based on https://visualstudiomagazine.com/articles/2021/06/23/logistic-regression-pytorch.aspx
"""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss


class LogisticRegression:
    def __init__(self, C, max_iter, verbose, random_state, **kwargs):
        self.C = C
        # --- CHANGE: Initialize loss_func later, after we have the weights ---
        # self.loss_func = torch.nn.CrossEntropyLoss() 
        self.loss_func: CrossEntropyLoss | None = None 
        self.max_iter = max_iter
        self.random_state = random_state
        print(self.random_state)
        self.logreg = None
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- CHANGE: Store the class_weight parameter ---
        self.class_weight = kwargs.get('class_weight', None)

    def compute_loss(self, feats, labels):
        # The loss function now correctly weights the classes
        loss = self.loss_func(feats, labels)
        wreg = 0.5 * self.logreg.weight.norm(p=2)
        return loss.mean() + (1.0 / self.C) * wreg

    def predict_proba(self, feats):
        assert self.logreg is not None, "Need to fit first before predicting probs"
        return self.logreg(feats.to(self.device)).softmax(dim=-1)

    def fit(self, feats, labels):
        feat_dim = feats.shape[1]
        num_classes = len(torch.unique(labels))

        # set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # --- CHANGE: Calculate and apply class weights ---
        weights = None
        if self.class_weight == 'balanced':
            print("Calculating balanced class weights for custom Logistic Regression.")
            # Use scikit-learn to compute weights, which is robust
            # We need to compute this on the CPU with numpy arrays
            labels_np = labels.cpu().numpy()
            classes = np.unique(labels_np)
            
            # This computes weights as: n_samples / (n_classes * np.bincount(y))
            class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=labels_np)
            
            # Convert weights to a PyTorch tensor and move to the correct device
            weights = torch.tensor(class_weights_np, dtype=torch.float32).to(self.device)
            print(f"Computed weights: {weights}")

        # Initialize the loss function with the calculated weights
        # If no weights, it behaves like the original implementation
        self.loss_func = torch.nn.CrossEntropyLoss(weight=weights)
        # --- END OF CHANGE ---

        self.logreg = torch.nn.Linear(feat_dim, num_classes, bias=True)

        # move everything to CUDA .. otherwise why are we even doing this?!
        self.logreg = self.logreg.to(self.device)
        feats = feats.to(self.device)
        labels = labels.to(self.device)

        # define the optimizer
        opt = torch.optim.LBFGS(
            self.logreg.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.max_iter,
        )
        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(Before Training) Loss: {loss:.3f}")

        def loss_closure():
            opt.zero_grad()
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            loss.backward()
            return loss

        opt.step(loss_closure)  # get loss, use to update wts

        if self.verbose:
            pred = self.logreg(feats)
            loss = self.compute_loss(pred, labels)
            print(f"(After Training) Loss: {loss:.3f}")