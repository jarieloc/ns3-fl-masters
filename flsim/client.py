# flsim/client.py
import logging
import torch
import random
import os


class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):
        self.client_id = client_id
        # Initialize a large loss so new/idle clients are sampled first
        self.loss = 10.0
        # DP config placeholder; may be set by the server each round
        self.dp = None

    def __repr__(self):
        return f'Client #{self.client_id}'

    # ------- Non-IID knobs -------
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):
        self.shard = shard

    # ------- Server I/O (simulated) -------
    def download(self, argv):
        try:
            return argv.copy()
        except Exception:
            return argv

    def upload(self, argv):
        try:
            return argv.copy()
        except Exception:
            return argv

    # ------- Data -------
    def set_data(self, data, config):
        # Extract from config
        self.do_test = config.clients.do_test
        self.test_partition = config.clients.test_partition

        # Download data partition
        self.data = self.download(data)

        # Split train/test if requested
        if self.do_test:
            n_train = int(len(self.data) * (1 - self.test_partition))
            self.trainset = self.data[:n_train]
            self.testset = self.data[n_train:]
        else:
            self.trainset = self.data

    # ------- Link / delay model -------
    def set_link(self, config):
        # Gaussian link speed (KBytes/s) bounds
        self.speed_min = config.link.min
        self.speed_max = config.link.max
        self.speed_mean = random.uniform(self.speed_min, self.speed_max)
        self.speed_std = config.link.std

        # Model size (KB)
        self.model_size = config.model.size
        # Estimated delay for scheduling
        self.est_delay = self.model_size / self.speed_mean

    def set_delay(self):
        # Draw a speed for this run and clamp to bounds
        link_speed = random.normalvariate(self.speed_mean, self.speed_std)
        link_speed = max(min(link_speed, self.speed_max), self.speed_min)
        # Upload delay in seconds
        self.delay = self.model_size / link_speed

    # ------- FL configuration -------
    def configure(self, config):
        import fl_model  # pylint: disable=import-error

        self.model_path = config.paths.model
        cfg = self.download(config)

        # FL task
        self.task = cfg.fl.task
        self.epochs = cfg.fl.epochs
        self.batch_size = cfg.fl.batch_size

        # Load latest global model
        path = os.path.join(self.model_path, 'global')
        self.model = fl_model.Net()
        # Always map to CPU here; device decisions happen inside fl_model.train/test
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()

        # Optimizer
        self.optimizer = fl_model.get_optimizer(self.model)

        # DP config (keep any value already set by the server)
        self.dp = getattr(self, "dp", None) or getattr(cfg, "dp", None) \
                  or getattr(getattr(cfg, "fl", object()), "dp", None)

    def async_configure(self, config, download_time):
        import fl_model  # pylint: disable=import-error

        self.model_path = config.paths.model
        cfg = self.download(config)

        # FL task
        self.task = cfg.fl.task
        self.epochs = cfg.fl.epochs
        self.batch_size = cfg.fl.batch_size

        # Load snapshot corresponding to network download time
        path = os.path.join(self.model_path, f'global_{download_time}')
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
        logging.info('Load global model: %s', path)

        # Optimizer
        self.optimizer = fl_model.get_optimizer(self.model)

        # DP config (prefer what the server injected this round)
        self.dp = getattr(self, "dp", None) or getattr(cfg, "dp", None) \
                  or getattr(getattr(cfg, "fl", object()), "dp", None)

    # ------- FL phases -------
    def run(self, reg=None):
        if self.task == "train":
            self.train(reg)
        else:
            raise NotImplementedError(f"Unsupported task: {self.task}")

    def get_report(self):
        return self.upload(self.report)

    # ------- ML tasks -------
    def train(self, reg=None):
        import fl_model  # pylint: disable=import-error

        logging.info('Training on client #%d, mean delay %ss',
                     self.client_id, self.delay)

        trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
        dp_cfg = getattr(self, "dp", None)
        self.loss = fl_model.train(
            self.model,
            trainloader,
            self.optimizer,
            self.epochs,
            reg=reg,
            dp=dp_cfg
        )

        # Extract model weights
        weights = fl_model.extract_weights(self.model)

        # Build report
        self.report = Report(self)
        self.report.weights = weights
        self.report.loss = self.loss
        self.report.delay = self.delay

        # Optional test
        if self.do_test:
            testloader = fl_model.get_testloader(self.testset, 1000)
            self.report.accuracy = fl_model.test(self.model, testloader)

    def test(self):
        raise NotImplementedError


class Report(object):
    """Federated learning client report."""

    def __init__(self, client):
        self.client_id = client.client_id
        self.num_samples = len(client.data)


# # flsim/client.py
# import logging
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import os
# import time


# class Client(object):
#     """Simulated federated learning client."""

#     def __init__(self, client_id):
#         self.client_id = client_id
#         # Initialize a large loss so new/idle clients are sampled first
#         self.loss = 10.0
#         self.dp = None

#     def __repr__(self):
#         return f'Client #{self.client_id}'

#     # ------- Non-IID knobs -------
#     def set_bias(self, pref, bias):
#         self.pref = pref
#         self.bias = bias

#     def set_shard(self, shard):
#         self.shard = shard

#     # ------- Server I/O (simulated) -------
#     def download(self, argv):
#         try:
#             return argv.copy()
#         except Exception:
#             return argv

#     def upload(self, argv):
#         try:
#             return argv.copy()
#         except Exception:
#             return argv

#     # ------- Data -------
#     def set_data(self, data, config):
#         # Extract from config
#         self.do_test = config.clients.do_test
#         self.test_partition = config.clients.test_partition

#         # Download data partition
#         self.data = self.download(data)

#         # Split train/test if requested
#         if self.do_test:
#             n_train = int(len(self.data) * (1 - self.test_partition))
#             self.trainset = self.data[:n_train]
#             self.testset = self.data[n_train:]
#         else:
#             self.trainset = self.data

#     # ------- Link / delay model -------
#     def set_link(self, config):
#         # Gaussian link speed (KBytes/s) bounds
#         self.speed_min = config.link.min
#         self.speed_max = config.link.max
#         self.speed_mean = random.uniform(self.speed_min, self.speed_max)
#         self.speed_std = config.link.std

#         # Model size (KB)
#         self.model_size = config.model.size
#         # Estimated delay for scheduling
#         self.est_delay = self.model_size / self.speed_mean

#     def set_delay(self):
#         # Draw a speed for this run and clamp to bounds
#         link_speed = random.normalvariate(self.speed_mean, self.speed_std)
#         link_speed = max(min(link_speed, self.speed_max), self.speed_min)
#         # Upload delay in seconds
#         self.delay = self.model_size / link_speed

#     # ------- FL configuration -------
#     def configure(self, config):
#         import fl_model  # pylint: disable=import-error

#         self.model_path = config.paths.model
#         cfg = self.download(config)

#         # FL task
#         self.task = cfg.fl.task
#         self.epochs = cfg.fl.epochs
#         self.batch_size = cfg.fl.batch_size

#         # Load latest global model
#         path = os.path.join(self.model_path, 'global')
#         self.model = fl_model.Net()
#         self.model.load_state_dict(torch.load(path, map_location='cpu'))
#         self.model.eval()

#         # Optimizer
#         self.optimizer = fl_model.get_optimizer(self.model)

#     def async_configure(self, config, download_time):
#         import fl_model  # pylint: disable=import-error

#         self.model_path = config.paths.model
#         cfg = self.download(config)

#         # FL task
#         self.task = cfg.fl.task
#         self.epochs = cfg.fl.epochs
#         self.batch_size = cfg.fl.batch_size

#         # Optionally pass DP block down to client (if present in config)
#         self.dp = getattr(cfg, "dp", None)
#         if self.dp is None:
#             self.dp = getattr(getattr(cfg, "fl", object()), "dp", None)

#         # Load snapshot corresponding to network download time
#         path = os.path.join(self.model_path, f'global_{download_time}')
#         self.model = fl_model.Net()
#         self.model.load_state_dict(torch.load(path, map_location='cpu'))
#         self.model.eval()
#         logging.info('Load global model: %s', path)

#         # Optimizer
#         self.optimizer = fl_model.get_optimizer(self.model)

#     # ------- FL phases -------
#     def run(self, reg=None):
#         if self.task == "train":
#             self.train(reg)
#         else:
#             raise NotImplementedError(f"Unsupported task: {self.task}")

#     def get_report(self):
#         return self.upload(self.report)

#     # ------- ML tasks -------
#     def train(self, reg=None):
#         import fl_model  # pylint: disable=import-error

#         logging.info('Training on client #%d, mean delay %ss',
#                      self.client_id, self.delay)

#         trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
#         dp_cfg = getattr(self, "dp", None)
#         self.loss = fl_model.train(
#             self.model,
#             trainloader,
#             self.optimizer,   # not a bare "optimizer"
#             self.epochs,      # not a bare "epochs"
#             reg=reg,
#             dp=dp_cfg
#         )

#         # Extract model weights
#         weights = fl_model.extract_weights(self.model)

#         # Build report
#         self.report = Report(self)
#         self.report.weights = weights
#         self.report.loss = self.loss
#         self.report.delay = self.delay

#         # Optional test
#         if self.do_test:
#             testloader = fl_model.get_testloader(self.testset, 1000)
#             self.report.accuracy = fl_model.test(self.model, testloader)

#     def test(self):
#         raise NotImplementedError


# class Report(object):
#     """Federated learning client report."""

#     def __init__(self, client):
#         self.client_id = client.client_id
#         self.num_samples = len(client.data)


# # import logging
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import random
# # import os
# # import time

# # class Client(object):
# #     """Simulated federated learning client."""

# #     def __init__(self, client_id):
# #         self.client_id = client_id
# #         self.loss = 10.0  # Set a big number for init loss
# #                           # to first select clients that haven't been selected

# #     def __repr__(self):
# #         #return 'Client #{}: {} samples in labels: {}'.format(
# #         #    self.client_id, len(self.data), set([label for _, label in self.data]))
# #         return 'Client #{}'.format(self.client_id)

# #     # Set non-IID data configurations
# #     def set_bias(self, pref, bias):
# #         self.pref = pref
# #         self.bias = bias

# #     def set_shard(self, shard):
# #         self.shard = shard

# #     # Server interactions
# #     def download(self, argv):
# #         # Download from the server.
# #         try:
# #             return argv.copy()
# #         except:
# #             return argv

# #     def upload(self, argv):
# #         # Upload to the server
# #         try:
# #             return argv.copy()
# #         except:
# #             return argv

# #     # Federated learning phases
# #     def set_data(self, data, config):
# #         # Extract from config
# #         do_test = self.do_test = config.clients.do_test
# #         test_partition = self.test_partition = config.clients.test_partition

# #         # Download data
# #         self.data = self.download(data)

# #         # Extract trainset, testset (if applicable)
# #         data = self.data
# #         if do_test:  # Partition for testset if applicable
# #             self.trainset = data[:int(len(data) * (1 - test_partition))]
# #             self.testset = data[int(len(data) * (1 - test_partition)):]
# #         else:
# #             self.trainset = data

# #     def set_link(self, config):
# #         # Set the Gaussian distribution for link speed in Kbytes
# #         self.speed_min = config.link.min
# #         self.speed_max = config.link.max
# #         self.speed_mean = random.uniform(self.speed_min, self.speed_max)
# #         self.speed_std = config.link.std

# #         # Set model size
# #         '''model_path = config.paths.model + '/global'
# #         if os.path.exists(model_path):
# #             self.model_size = os.path.getsize(model_path) / 1e3  # model size in Kbytes
# #         else:
# #             self.model_size = 1600  # estimated model size in Kbytes'''
# #         self.model_size = config.model.size
# #         # Set estimated delay
# #         self.est_delay = self.model_size / self.speed_mean

# #     def set_delay(self):
# #         # Set the link speed and delay for the upcoming run
# #         link_speed = random.normalvariate(self.speed_mean, self.speed_std)
# #         link_speed = max(min(link_speed, self.speed_max), self.speed_min)
# #         self.delay = self.model_size / link_speed  # upload delay in sec

# #     def configure(self, config):
# #         import fl_model  # pylint: disable=import-error

# #         # Extract from config
# #         model_path = self.model_path = config.paths.model

# #         # Download from server
# #         config = self.download(config)

# #         # Extract machine learning task from config
# #         self.task = config.fl.task
# #         self.epochs = config.fl.epochs
# #         self.batch_size = config.fl.batch_size

# #         # Download most recent global model
# #         path = model_path + '/global'
# #         self.model = fl_model.Net()
# #         self.model.load_state_dict(torch.load(path))
# #         self.model.eval()

# #         # Create optimizer
# #         self.optimizer = fl_model.get_optimizer(self.model)

# #     def async_configure(self, config, download_time):
# #         import fl_model  # pylint: disable=import-error

# #         # Extract from config
# #         model_path = self.model_path = config.paths.model

# #         # Download from server
# #         config = self.download(config)

# #         # Extract machine learning task from config
# #         self.task = config.fl.task
# #         self.epochs = config.fl.epochs
# #         self.batch_size = config.fl.batch_size

# #         # Download most recent global model
# #         path = model_path + '/global_' + '{}'.format(download_time)
# #         self.model = fl_model.Net()
# #         self.model.load_state_dict(torch.load(path))
# #         self.model.eval()
# #         logging.info('Load global model: {}'.format(path))

# #         # Create optimizer
# #         self.optimizer = fl_model.get_optimizer(self.model)


# #     def run(self, reg=None):
# #         # Perform federated learning task
# #         {
# #             "train": self.train(reg)
# #         }[self.task]

# #     def get_report(self):
# #         # Report results to server.
# #         return self.upload(self.report)

# #     # Machine learning tasks
# #     def train(self, reg=None):
# #         import fl_model  # pylint: disable=import-error
# #         import torch
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         self.model.to(device)

# #         logging.info('Training on client #{}, mean delay {}s'.format(
# #             self.client_id, self.delay))

# #         # Perform model training
# #         trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
# #         dp_cfg = getattr(self, "dp", None)  # <- use the field we set in asyncServer
# #         self.loss = fl_model.train(self.model, trainloader, optimizer, epochs, reg=reg, dp=dp_cfg)


# #         # self.loss = fl_model.train(self.model, trainloader,
# #         #                self.optimizer, self.epochs, reg) #not reg


# #         # Extract model weights and biases
# #         weights = fl_model.extract_weights(self.model)

# #         # Generate report for server
# #         self.report = Report(self)
# #         self.report.weights = weights
# #         self.report.loss = self.loss
# #         self.report.delay = self.delay

# #         # Perform model testing if applicable
# #         if self.do_test:
# #             testloader = fl_model.get_testloader(self.testset, 1000)
# #             self.report.accuracy = fl_model.test(self.model, testloader)

# #     def test(self):
# #         # Perform model testing
# #         raise NotImplementedError


# # class Report(object):
# #     """Federated learning client report."""

# #     def __init__(self, client):
# #         self.client_id = client.client_id
# #         self.num_samples = len(client.data)
