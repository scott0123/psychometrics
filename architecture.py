'''
First draft by Scott Liu @ 2020 Feb
'''

# Import statements
import math
import torch
import numpy as np
from metrics import get_f1
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class ExperimentalAdditiveFactorModel(nn.Module):
    def __init__(self, n_users, n_questions, Q):
        '''
        n_users: number of users
        n_questions: number of questions
        Q: Q matrix for the knowledge components of each question (np array)
        '''
        super().__init__()
        self.config = {
            "n_users": n_users,
            "n_questions": n_questions,
        }
        n_KCs = Q.shape[0]
        assert n_questions == Q.shape[1]
        # create and initialize the student-topic capacity vector
        self.alpha = Parameter(torch.randn(n_users, n_KCs))
        nn.init.xavier_uniform_(self.alpha)
        # create and initialize the KC difficulty vector
        self.beta = Parameter(torch.randn(n_KCs) / 2)
        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        if isinstance(Q, torch.Tensor):
            self.config["Q"] = Q.to(self.device)
        else:
            self.config["Q"] = torch.from_numpy(Q).to(self.device)


    def forward(self, users, questions):
        '''
        The inputs to the forward function should be:
            users: list of user indicies e.g. [0, 0, 0, 1, 1, 1, 2, ...]
            questions: list of question indices e.g. [0, 1, 2, 0, 1, 2, 0, ...]

        Note: these have to be of the same length
        '''
        Q = self.config["Q"]
        i = torch.LongTensor(users).to(self.device)
        j = torch.LongTensor(questions).to(self.device)
        sum1 = torch.sum(self.alpha[i] * Q[:,j].T, dim=1)
        sum2 = self.beta @ Q[:,j]
        #factors = self.theta[i] + self.phi[j] + sum1 + sum2
        factors = sum1 + sum2
        return torch.sigmoid(factors)

    def fit(self, users, questions, corrects, lr=0.001, epochs=10):
        '''
        users: list of user indicies e.g. [0, 0, 0, 1, 1, 1, 2, ...]
        questions: list of question indices e.g. [0, 1, 2, 0, 1, 2, 0, ...]
        corrects: list of whether the user's answer to the
            question was correct or not e.g. [0, 1, 1, 0, 1, 1, 1, ...]
        lr: learning rate
        epochs: number of epochs to train
        '''
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            # Clear the gradients of parameters
            optimizer.zero_grad()
            # Perform forward pass to get model outputs
            y_hats = self.forward(users, questions)
            # True labels
            ys = torch.tensor(corrects, dtype=torch.float)
            # Calculate the loss
            loss = loss_fn(y_hats, ys)
            predicted = (y_hats > 0.5).float()
            accuracy = (predicted == ys).sum().item() / len(ys)
            f1, precision, recall = get_f1(ys.cpu().detach().numpy(), predicted.cpu().detach().numpy())
            # Call `backward()` on `loss` for back-propagation to compute
            # gradients w.r.t. model parameters
            loss.backward()
            # Perform one step of parameter update using the newly-computed gradients
            optimizer.step()
            print(f'Epoch {epoch+1}, loss={loss.item():.4f}, acc={accuracy:.4f}, f1={f1:.4f}')

    def auto_fit(self, users, questions, corrects,
                 val_users, val_questions, val_corrects,
                 lr=1e-3, reg=1e-2, patience=10):
        '''
        users: list of user indicies e.g. [0, 0, 0, 1, 1, 1, 2, ...]
        questions: list of question indices e.g. [0, 1, 2, 0, 1, 2, 0, ...]
        corrects: list of whether the user's answer to the
            question was correct or not e.g. [0, 1, 1, 0, 1, 1, 1, ...]
        val_*: same data but from the validation set
        lr: learning rate
        reg: weight decay coefficient (similar to L2 penalty)
        patience: number of epochs to continue evaluating even after f1 not increasing
        '''
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        loss_fn = nn.MSELoss()
        epoch = 0
        frustration = 0
        best_f1 = 0
        ys = torch.tensor(corrects, dtype=torch.float).to(self.device)
        val_ys = torch.tensor(val_corrects, dtype=torch.float).to(self.device)
        while True:
            # Clear the gradients of parameters
            optimizer.zero_grad()
            # Perform forward pass to get model outputs
            y_hats = self.forward(users, questions)
            # True labels
            # Calculate the loss
            loss = loss_fn(y_hats, ys)
            predicted = (y_hats > 0.5).float()
            accuracy = (predicted == ys).sum().item() / len(ys)
            f1, precision, recall = get_f1(ys.cpu().detach().numpy(), predicted.cpu().detach().numpy())
            val_y_hats = self.forward(val_users, val_questions)
            val_predicted = (val_y_hats > 0.5).float()
            val_accuracy = (val_predicted == val_ys).sum().item() / len(val_ys)
            val_f1, _, _ = get_f1(val_ys.cpu().detach().numpy(), val_predicted.cpu().detach().numpy())
            # Call `backward()` on `loss` for back-propagation to compute
            # gradients w.r.t. model parameters
            loss.backward()
            # Perform one step of parameter update using the newly-computed gradients
            optimizer.step()
            print(f'Epoch {epoch+1}, loss={loss.item():.4f}, acc={accuracy:.4f}, f1={f1:.4f}, val_acc={val_accuracy:.4f}, val_f1={val_f1:.4f}')
            if val_f1 > best_f1:
                best_f1 = val_f1
                frustration = 0
            else:
                frustration += 1
            if frustration > patience:
                break
            epoch += 1

    def predict(self, users, questions):
        self.eval()
        return self.forward(users, questions).cpu().detach().numpy()

    def predict_and_eval(self, users, questions, corrects):
        self.eval()
        y_hats = self.forward(users, questions)
        ys = torch.tensor(corrects, dtype=torch.float).to(self.device)
        predicted = (y_hats > 0.5).float()
        accuracy = (predicted == ys).sum().item() / len(ys)
        return accuracy

    def save(self, model_path):
        model_state = {
            "state_dict": self.state_dict(),
            "config":  self.config
        }
        torch.save(model_state, model_path)

    @classmethod
    def load(cls, model_path):
        model_state = torch.load(str(model_path), map_location=lambda storage, loc: storage)
        args = model_state["config"]
        model = cls(**args)
        model.load_state_dict(model_state["state_dict"])
        # Use GPU if available
        if torch.cuda.is_available():
            model.device = torch.device("cuda")
        else:
            model.device = torch.device("cpu")
        model.to(model.device)
        return model


class BackgroundRelationModel(nn.Module):
    def __init__(self, n_bg, n_ml):
        '''
        n_bg: number of background tags
        n_ml: number of ml tags
        '''
        super().__init__()
        self.config = {
            "n_bg": n_bg,
            "n_ml": n_ml,
        }
        # create and initialize the relation matrix
        self.W = Parameter(torch.randn(n_bg, n_ml))
        self.b = Parameter(torch.randn(n_ml) / 2)
        nn.init.xavier_uniform_(self.W)
        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)


    def forward(self, alpha_bg):
        '''
        The inputs to the forward function should be:
            the alpha matrix (np array) with columns corresponding to the background tags
            it should have the shape (batch_size, n_bg)
        '''
        alpha_bg_tensor = torch.tensor(alpha_bg).to(self.device)
        return alpha_bg_tensor @ self.W + self.b

    def fit(self, alpha_bg, alpha_ml, lr=0.001, epochs=10):
        '''
        alpha_bg: the alpha matrix (np array) with columns corresponding to the background tags
        alpha_ml: the alpha matrix (np array) with columns corresponding to the ML tags
        lr: learning rate
        epochs: number of epochs to train
        '''
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        ys = torch.tensor(alpha_ml).to(self.device)
        for epoch in range(epochs):
            # Clear the gradients of parameters
            optimizer.zero_grad()
            # Perform forward pass to get model outputs
            y_hats = self.forward(alpha_bg)
            # Calculate the loss
            loss = loss_fn(y_hats, ys)
            # Call `backward()` on `loss` for back-propagation to compute
            # gradients w.r.t. model parameters
            loss.backward()
            # Perform one step of parameter update using the newly-computed gradients
            optimizer.step()
            print(f'Epoch {epoch+1}, loss={loss.item():.4f}')

    def auto_fit(self, alpha_bg, alpha_ml, val_alpha_bg, val_alpha_ml,
                 lr=1e-3, reg=1e-2, patience=10):
        '''
        alpha_bg: the alpha matrix (np array) with columns corresponding to the background tags
        alpha_ml: the alpha matrix (np array) with columns corresponding to the ML tags
        val_*: same data but from the validation set
        lr: learning rate
        reg: weight decay coefficient (similar to L2 penalty)
        patience: number of epochs to continue evaluating even after loss not decreasing
        '''
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        loss_fn = nn.MSELoss()
        epoch = 0
        frustration = 0
        best_loss = np.inf
        ys = torch.tensor(alpha_ml).to(self.device)
        val_ys = torch.tensor(val_alpha_ml).to(self.device)
        while True:
            # Clear the gradients of parameters
            optimizer.zero_grad()
            # Perform forward pass to get model outputs
            y_hats = self.forward(alpha_bg)
            # Calculate the loss
            loss = loss_fn(y_hats, ys)
            # Call `backward()` on `loss` for back-propagation to compute
            # gradients w.r.t. model parameters
            loss.backward()
            # Perform one step of parameter update using the newly-computed gradients
            optimizer.step()
            val_y_hats = self.forward(val_alpha_bg)
            val_loss = loss_fn(val_y_hats, val_ys)
            print(f'Epoch {epoch+1}, loss={loss.item():.4f}, val_loss={val_loss.item():.4f}')
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                frustration = 0
            else:
                frustration += 1
            if frustration > patience:
                break
            epoch += 1

    def predict(self, alpha_bg):
        self.eval()
        return self.forward(alpha_bg).cpu().detach().numpy()

    def save(self, model_path):
        model_state = {
            "state_dict": self.state_dict(),
            "config":  self.config
        }
        torch.save(model_state, model_path)

    @classmethod
    def load(cls, model_path):
        model_state = torch.load(str(model_path), map_location=lambda storage, loc: storage)
        args = model_state["config"]
        model = cls(**args)
        model.load_state_dict(model_state["state_dict"])
        # Use GPU if available
        if torch.cuda.is_available():
            model.device = torch.device("cuda")
        else:
            model.device = torch.device("cpu")
        model.to(model.device)
        return model



def main():
    pass


if __name__ == "__main__":
    main()
