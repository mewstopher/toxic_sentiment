from toxic_sentiment.data_processors.functions import train_test_sampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from decouple import config, UndefinedValueError
import matplotlib as plt
import numpy as np
import datetime
import logging
import torch


class Session:

    losses = {}
    accuracies = {}

    def __init__(self, model, loss=torch.nn.BCELoss, optimizer=None,
                 learning_rate=0.005,save=False, load_model=False,
                 custom_input=False):
        self.logger = logging.getLogger(__name__)
        self.logger.debug('{} entered'.format(__name__))
        self.writer = SummaryWriter()
        self.user_input(custom_input)
        self.save_path = self.save_choice(save)
        self.device = self.get_device()
        self.model = model
        self.optimizer = self.select_optimizer(optimizer, learning_rate)
        self.load_saved_model(load_model)
        self.loss = loss
        self.save = save
        self.count = 0
        self.total = 0
        self.correct = 0

    def load_saved_model(self, load_model: bool):
        """
        check if load model is True. If its true, then load model from path.
        """

        if load_model:
            try:
                model_path = config('MODEL_PATH')
                self.logger.info('loading saved model from path: {}'.format(model_path))
                self.model.load_state_dict(torch.load(model_path))
                self.model.train()
            except UndefinedValueError:
                self.logger.info('You\"ve selected to load a model but have not defined the path '
                                 'in the .env file. See env.template')
        else:
            self.logger.info('no model specified. Training from scratch')

    def user_input(self, custom_input):
        """
        decide whether or not to use custom user input
        """
        if not custom_input:
            self.logger.info('You have selected to not use '
                             'custom input. the following choices '
                             'have been made for you: '
                             'model: {}'
                             'loss: {}'
                             'learning rate: '
                             'tran/val/test split: {}, {}, {}'
                             )
            self.logger.info('To change this, use the --custom option')
        else:
            self.logger.info('using .ENV for hyperparameters.')

    def get_device(self):
        """
        decide whether to use cpu or gpu based on availiability
        """
        if torch.cuda.is_available():
            device = "cuda:0"
            self.logger.info('Using GPU')
        else:
            device = "cpu"
            self.logger.info('Using CPU')
        return device

    def select_optimizer(self, optimizer, learning_rate):
        if not optimizer:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def save_choice(self, save: bool) -> str:
        """
        Use save option to save model.
        Check for save path existence in .ENV file
        """
        save_path = None
        if save:
            try:
                save_path = config('MODEL_SAVE_PATH')
                self.logger.info("saving model at: {}".format(save_path))
            except UndefinedValueError:
                self.logger.error('model path not defined. Use .ENV file template')
        else:
            self.logger.info("you have chosen not to save the model")
        return save_path

    def train_epoch(self, train_dataloader, epoch, loss):
        for data_sample in train_dataloader:
            for i in range(len(data_sample)):
                data_sample[i] = data_sample[i].to(self.device)
            self.count += 1
            self.model.zero_grad()
            self.writer.add_graph(self.model, data_sample[0])
            out = self.model(data_sample[0])
            sample_loss = loss(out, data_sample[1].float())

            sample_loss.backward()
            self.optimizer.step()
            self.losses[epoch].append(sample_loss.item())

            predicted = torch.round(out)
            self.total += data_sample[1].size(0) * data_sample[1].size(1)
            self.correct += (predicted == data_sample[1]).sum().item()
            accuracy = self.correct / self.total
            self.accuracies[epoch].append(accuracy)

            if self.count % 10 == 0:
                self.logger.info("loss: {} (at iteration {})".format(np.mean(self.losses[epoch]), self.count))
                self.logger.info("accuracy: {} (at iteration {})".format(np.mean(self.accuracies[epoch]), self.count))
        self.writer.add_scalar('Training loss', sample_loss.item(), epoch)
        self.writer.add_scalar('Training Accuracy', accuracy, epoch)

    def train(self, dataset, batch_size, epochs):
        sampled_data = train_test_sampler(dataset,
                                          train_split=.8,
                                          val_split=.1,
                                          test_split=.1)
        train_dataloader = self.create_dataloader(dataset, batch_size, sampler=sampled_data[0])
        self.logger.info("beginning to train the machine")
        loss = self.loss()
        for epoch in range(epochs):
            self.losses[epoch] = []
            self.accuracies[epoch] = []
            self.train_epoch(train_dataloader, epoch, loss)
            self.logger.info("Average training loss for epoch {}: {}".format(epoch, np.mean(self.losses[epoch])))
            if self.save_path:
                torch.save(self.model.state_dict(), self.save_path + str(datetime.datetime.now()).split()[0])

    def create_dataloader(self, dataset, batch_size, sampler):
        """
        create dataloaders
        """
        data_loader = DataLoader(dataset, batch_size, sampler=sampler)
        return data_loader

    def print_losses(self):
        loss_list = [i for epoch in self.losses for i in self.losses[epoch]]
        return plt.plot(loss_list)

    def plot_accuracies(self):
        accuracy_list = [i for epoch in self.accuracies for i in self.accuracies[epoch]]
        return plt.plot(accuracy_list)

    def run(self, dataset, epochs):
        try:
            batch_size = config('BATCH_SIZE')
        except UndefinedValueError:
            batch_size = 32
        try:
            self.train(dataset, batch_size, epochs)

        except KeyboardInterrupt:
            print('\n' * 8)
            if self.save:
                torch.save(self.model.state_dict(), self.save_path + str(datetime.datetime.now()).split()[0])
