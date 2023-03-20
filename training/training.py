import os
import numpy as np
import torch
import pandas as pd
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def pandas_from_mat(input_mat):
    input_mat = np.asarray(input_mat)
    dataframe = pd.DataFrame({
        'TP': input_mat[:, 0, 0],
        'FP': input_mat[:, 0, 1],
        'FN': input_mat[:, 1, 0],
        'TN': input_mat[:, 1, 1],
    })
    dataframe.index += 1
    return dataframe


def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, n_epochs, loss_function, scheduler,
          save_path):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    train_specificities = []
    val_specificities = []
    test_specificities = []
    # Confusion matrix = [[true_positive, false_positive], [false_negative, true_negative]]
    train_mats = []
    val_mats = []
    test_mats = []

    print(torch.version.cuda())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    for epoch in range(n_epochs):  # do n_epochs
        model.train()  # model se koristi za treniranje
        losses = []
        conf_mat = [[0, 0], [0, 0]]

        for variables, ys in train_dataloader:  # za svaki skup ulaznih podataka
            variables = variables.to(device)
            ys = ys.to(device)
            output = model(variables)  # predikcija na osnovu variables tj ulaza

            optimizer.zero_grad()  # u PyTorch moramo da postavimo gradijente na 0 pre propagacije unazad
            loss = loss_function(output, ys)
            loss.backward()  # loss se propagira pozadi i racuna koliko bi trebalo da se updateuje sve od tezina
            optimizer.step()  # radi update parametara, update tezina

            losses.append(loss.item())  # u niz loss za batcheve se doda loss za batch

            correct = output.argmax(1) == ys
            positive = ys == 1
            negative = ys == 0
            incorrect = output.argmax(1) != ys
            conf_mat[0][0] += torch.sum(correct * positive).item()
            conf_mat[0][1] += torch.sum(incorrect * negative).item()
            conf_mat[1][0] += torch.sum(incorrect * positive).item()
            conf_mat[1][1] += torch.sum(correct * negative).item()

        train_losses.append(np.mean(np.array(losses)))
        train_accuracies.append(100.0 * (conf_mat[0][0] + conf_mat[1][1]) / len(train_dataloader.dataset))
        train_specificities.append(100 * conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]))
        train_mats.append(conf_mat)

        # Valuation
        model.eval()  # model in valuation mode
        losses = []
        conf_mat = [[0, 0], [0, 0]]
        with torch.no_grad():  # iskljuci racunanje gradijenta, ne treniramo
            # torch.no_grad smanjuje zauzece memorije
            for variables, ys in val_dataloader:
                variables = variables.to(device)
                ys = ys.to(device)
                output = model(variables)  # prosledimo ulaze, i dobijemo izlaz
                loss = loss_function(output,
                                     ys)  # sta smo dobili kao output i vrednosti koje su tacne, pa izraunaj koliko je lose

                losses.append(loss.item())  # u niz loss za batcheve se doda loss za batch

                correct = output.argmax(1) == ys
                positive = ys == 1
                negative = ys == 0
                incorrect = output.argmax(1) != ys
                conf_mat[0][0] += torch.sum(correct * positive).item()
                conf_mat[0][1] += torch.sum(incorrect * negative).item()
                conf_mat[1][0] += torch.sum(incorrect * positive).item()
                conf_mat[1][1] += torch.sum(correct * negative).item()

        # Save validation results
        val_losses.append(np.mean(np.array(losses)))
        val_accuracies.append(100.0 * (conf_mat[0][0] + conf_mat[1][1]) / len(val_dataloader.dataset))
        val_specificities.append(100 * conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]))
        val_mats.append(conf_mat)

        # Test
        conf_mat = [[0, 0], [0, 0]]
        with torch.no_grad():  # iskljuci racunanje gradijenta, ne treniramo
            # torch.no_grad smanjuje zauzece memorije
            for variables, ys in test_dataloader:
                variables = variables.to(device)
                ys = ys.to(device)
                output = model(variables)  # prosledimo ulaze, i dobijemo izlaz
                correct = output.argmax(1) == ys
                positive = ys == 1
                negative = ys == 0
                incorrect = output.argmax(1) != ys
                conf_mat[0][0] += torch.sum(correct * positive).item()
                conf_mat[0][1] += torch.sum(incorrect * negative).item()
                conf_mat[1][0] += torch.sum(incorrect * positive).item()
                conf_mat[1][1] += torch.sum(correct * negative).item()

        # Save test results
        test_accuracies.append(100.0 * (conf_mat[0][0] + conf_mat[1][1]) / len(test_dataloader.dataset))
        test_specificities.append(100 * conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]))
        test_mats.append(conf_mat)

        # Update learning rate
        scheduler.step()

        # Show update
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f},'
              ' val_accuracy: {:.4f}'.format(epoch + 1, n_epochs,
                                             train_losses[-1],
                                             train_accuracies[-1], val_losses[-1],
                                             val_accuracies[-1]))
        # Graph
        fig = make_subplots(rows=3, cols=2, specs=[[{"colspan": 2}, None],
                                                   [{"colspan": 2}, None],
                                                   [{"colspan": 2}, None]])
        fig.add_trace(go.Scatter(y=train_losses, x=np.arange(1, len(train_losses)), mode='lines',
                                 name='Train Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(y=val_losses, x=np.arange(1, len(val_losses)),mode='lines',
                                 name='Val Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(y=train_accuracies, x=np.arange(1, len(train_accuracies)),mode='lines',
                                 name='Train Acc'), row=2, col=1)
        fig.add_trace(go.Scatter(y=val_accuracies, x=np.arange(1, len(val_accuracies)),mode='lines',
                                 name='Val Acc'), row=2, col=1)
        fig.add_trace(go.Scatter(y=test_accuracies, x=np.arange(1, len(test_accuracies)),mode='lines',
                                 name='Test Acc'), row=2, col=1)
        fig.add_trace(go.Scatter(y=train_specificities, x=np.arange(1, len(train_specificities)),mode='lines',
                                 name='Train Spec'), row=3, col=1)
        fig.add_trace(go.Scatter(y=val_specificities, x=np.arange(1, len(val_specificities)),mode='lines',
                                 name='Val Spec'), row=3, col=1)
        fig.add_trace(go.Scatter(y=test_specificities, x=np.arange(1, len(test_specificities)),mode='lines',
                                 name='Test Spec'), row=3, col=1)

        fig.write_html('{}/graph_{}.html'.format(save_path, epoch+1))
        if epoch > 0:
            os.remove('{}/graph_{}.html'.format(save_path, epoch))
            os.rename('{}/model.pth'.format(save_path), '{}/model_1.pth'.format(save_path))
        torch.save(model.state_dict(), '{}/model.pth'.format(save_path))
        if epoch > 0:
            os.remove('{}/model_1.pth'.format(save_path))
        train_results = pandas_from_mat(train_mats)
        val_results = pandas_from_mat(val_mats)
        test_results = pandas_from_mat(test_mats)
        with pd.ExcelWriter('{}/results_{}.xlsx'.format(save_path, epoch+1)) as writer:
            train_results.to_excel(writer,
                                   sheet_name='Train',
                                   index=True,
                                   header=True)
            val_results.to_excel(writer,
                                 sheet_name='Validation',
                                 index=True,
                                 header=True)
            test_results.to_excel(writer,
                                  sheet_name='Test',
                                  index=True,
                                  header=True)
        if epoch > 0:
            os.remove('{}/results_{}.xlsx'.format(save_path, epoch))
    return model


if __name__ == '__main__':
    class LinearModel(nn.Module):

        def __init__(self, input_dim):
            super(LinearModel, self).__init__()
            self.fc1 = nn.Sequential(
                nn.Linear(input_dim, 32, bias=True),
                nn.ReLU(inplace=True))
            self.fc2 = nn.Sequential(
                nn.Linear(32, 2, bias=True))
            # nn.ReLU(inplace=True))

        def forward(self, nn_input):
            nn_input2 = self.fc1(nn_input)
            out = self.fc2(nn_input2)
            return out


    class TestDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            datapoint = self.data[index]
            target = int((0 if self.data[index] < 0.5 else 1))
            return torch.tensor(datapoint, dtype=torch.float32), target

        def __len__(self):
            return len(self.data)


    x = np.random.rand(1000, 1)
    net = LinearModel(input_dim=1)
    train_proc = 70
    val_proc = 20
    test_proc = 10
    batch_size = 32
    epochs = 100
    train_proc = int(len(x) * train_proc / 100)
    val_proc = int(len(x) * val_proc / 100)
    test_proc = int(len(x) * test_proc / 100)
    dataset_train = TestDataset(x[:train_proc])
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_validation = TestDataset(x[train_proc:(train_proc + val_proc)])
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
    dataset_test = TestDataset(x[(train_proc + val_proc):])
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam([
        {'params': net.parameters()}
    ], lr=0.05, weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_function = nn.CrossEntropyLoss()
    train(net, dataloader_train, dataloader_validation, dataloader_test, optimizer, epochs, loss_function, scheduler,
          './test_results')
