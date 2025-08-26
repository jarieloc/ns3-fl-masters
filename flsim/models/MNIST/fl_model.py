# flsim/models/MNIST/fl_model.py
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

import load_data


# ----- Training settings -----
lr = 0.01
momentum = 0.9
log_interval = 10
rou = 1
loss_thres = 0.01

# ----- Device -----
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for MNIST dataset."""

    # Extract MNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.MNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.testset = datasets.MNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):
    """
    Extract trainable weights without moving the live model off its device.
    """
    weights = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            weights.append((name, p.detach().clone().to('cpu')))
    return weights


def load_weights(model, weights):
    updated_state_dict = {name: w for name, w in weights}
    model.load_state_dict(updated_state_dict, strict=False)


def flatten_weights(weights):
    vecs = []
    for _, w in weights:
        vecs.append(w.detach().view(-1).cpu().numpy())
    return np.concatenate(vecs)


def _dp_log_eps(privacy_engine, dp_cfg, where: str):
    """
    Try to log epsilon from Opacus regardless of minor API differences.
    """
    try:
        delta = float(dp_cfg.get("delta", 1e-5))
        if hasattr(privacy_engine, "accountant"):
            eps = privacy_engine.accountant.get_epsilon(delta)
        else:
            eps = privacy_engine.get_epsilon(delta)
        logging.info(
            f"[DP] {where}: ε≈{eps:.2f}, δ={delta}, "
            f"σ={dp_cfg.get('noise_multiplier', 0.8)}, C={dp_cfg.get('max_grad_norm', 1.0)}"
        )
    except Exception as e:
        logging.debug(f"[DP] epsilon unavailable at {where}: {e}")


def train(model, trainloader, optimizer, epochs, reg=None, dp: Optional[dict] = None):
    """
    Train one client model. If dp is provided with dp['enable']=True,
    use Opacus to perform DP-SGD. Otherwise behaves as before.
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    # --- Enable DP-SGD (Opacus) if requested ---
    privacy_engine = None
    if dp and dp.get("enable", False):
        try:
            from opacus import PrivacyEngine
            noise_multiplier = float(dp.get("noise_multiplier", 0.8))
            max_grad_norm = float(dp.get("max_grad_norm", 1.0))
            secure_mode = bool(dp.get("secure_mode", False))
            accountant = dp.get("accountant", "rdp")

            privacy_engine = PrivacyEngine(accountant=accountant, secure_mode=secure_mode)
            model, optimizer, trainloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=trainloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            logging.info(f"[DP] Enabled: sigma={noise_multiplier}, C={max_grad_norm}, accountant={accountant}")
        except Exception as e:
            logging.warning(f"[DP] Failed to enable DP-SGD ({e}). Continuing without DP.")
            privacy_engine = None

    # --- Regularization snapshot (on same device as loss) ---
    if reg is not None:
        old_w_np = flatten_weights(extract_weights(model))
        old_weights = torch.from_numpy(old_w_np).to(device=device, dtype=torch.float32)
        mse_loss = nn.MSELoss(reduction='sum').to(device)

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(trainloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)

            if reg is not None:
                new_w_np = flatten_weights(extract_weights(model))
                new_weights = torch.from_numpy(new_w_np).to(device=device, dtype=torch.float32)
                l2_loss = rou / 2 * mse_loss(new_weights, old_weights)
                loss = loss + l2_loss

            loss.backward()
            optimizer.step()

            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(epoch, epochs, loss.item()))

            # Early stop if the model is already in good shape
            if loss.item() < loss_thres:
                if privacy_engine is not None:
                    _dp_log_eps(privacy_engine, dp, where="early-exit")
                return loss.item()

        # Per-epoch DP logging
        if privacy_engine is not None:
            _dp_log_eps(privacy_engine, dp, where=f"epoch {epoch}")

    # Final loss logging
    if reg is not None and 'l2_loss' in locals():
        logging.info('loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
    else:
        logging.info('loss: {}'.format(loss.item()))

    # Final DP logging
    if privacy_engine is not None:
        _dp_log_eps(privacy_engine, dp, where="final")

    return loss.item()


def test(model, testloader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = len(testloader.dataset)
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))
    return accuracy


# import load_data
# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# import numpy as np
# import time
# from typing import Optional



# # Training settings
# lr = 0.01
# momentum = 0.9
# log_interval = 10
# rou = 1
# loss_thres = 0.01

# # Cuda settings
# use_cuda = torch.cuda.is_available()
# device = torch.device(  # pylint: disable=no-member
#     'cuda' if use_cuda else 'cpu')


# class Generator(load_data.Generator):
#     """Generator for MNIST dataset."""

#     # Extract MNIST data using torchvision datasets
#     def read(self, path):
#         self.trainset = datasets.MNIST(
#             path, train=True, download=True, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     (0.1307,), (0.3081,))
#             ]))
#         self.testset = datasets.MNIST(
#             path, train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     (0.1307,), (0.3081,))
#             ]))
#         self.labels = list(self.trainset.classes)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# def get_optimizer(model):
#     return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# def get_trainloader(trainset, batch_size):
#     return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


# def get_testloader(testset, batch_size):
#     return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


# def extract_weights(model):
#     weights = []
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             # clone to CPU WITHOUT touching the model’s device
#             weights.append((name, p.detach().clone().to('cpu')))
#     return weights

# # def extract_weights(model):
# #     weights = []
# #     for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
# #         if weight.requires_grad:
# #             weights.append((name, weight.data))

# #     return weights


# def load_weights(model, weights):
#     updated_state_dict = {}
#     for name, weight in weights:
#         updated_state_dict[name] = weight

#     model.load_state_dict(updated_state_dict, strict=False)

# def flatten_weights(weights):
#     import numpy as np
#     vecs = []
#     for _, w in weights:
#         vecs.append(w.detach().view(-1).cpu().numpy())
#     return np.concatenate(vecs)

# # def flatten_weights(weights):
# #     # Flatten weights into vectors
# #     weight_vecs = []
# #     for _, weight in weights:
# #         weight_vecs.extend(weight.flatten().tolist())

# #     return np.array(weight_vecs)








# # def train(model, trainloader, optimizer, epochs, reg=None, dp: Optional[dict]=None):
# #     """
# #     If dp is provided and dp.get("enable", False) is True, wrap the model/optimizer/loader
# #     with Opacus. Otherwise behaves exactly as before.
# #     """
# #     model.to(device)
# #     model.train()
# #     criterion = nn.CrossEntropyLoss().to(device)

# #     # privacy_engine = None
# #     # if dp and dp.get("enable", False):
# #     #     # Lazy import to keep non-DP runs free of the dependency
# #     #     from opacus import PrivacyEngine
# #     #     noise_multiplier = float(dp.get("noise_multiplier", 0.8))
# #     #     max_grad_norm    = float(dp.get("max_grad_norm", 1.0))
# #     #     secure_mode      = bool(dp.get("secure_mode", False))

# #     #     privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=secure_mode)
# #     #     model, optimizer, trainloader = privacy_engine.make_private(
# #     #         module=model,
# #     #         optimizer=optimizer,
# #     #         data_loader=trainloader,
# #     #         noise_multiplier=noise_multiplier,
# #     #         max_grad_norm=max_grad_norm,
# #     #     )

# #     # Snapshot for regularization (does NOT move model off device)
# #     if reg is not None:
# #         old_weights = flatten_weights(extract_weights(model))
# #         old_weights = torch.from_numpy(old_weights)

# #     for epoch in range(1, epochs + 1):
# #         for batch_id, (image, label) in enumerate(trainloader):
# #             image, label = image.to(device), label.to(device)
# #             optimizer.zero_grad()
# #             output = model(image)
# #             loss = criterion(output, label)

# #             if reg is not None:
# #                 new_weights = flatten_weights(extract_weights(model))
# #                 new_weights = torch.from_numpy(new_weights)
# #                 mse_loss = nn.MSELoss(reduction='sum')
# #                 l2_loss = rou/2 * mse_loss(new_weights, old_weights)
# #                 l2_loss = l2_loss.to(torch.float32)
# #                 loss += l2_loss

# #             loss.backward()
# #             optimizer.step()

# #             if batch_id % log_interval == 0:
# #                 logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
# #                     epoch, epochs, loss.item()))

# #             if loss.item() < loss_thres:
# #                 if privacy_engine is not None:
# #                     delta = float(dp.get("delta", 1e-5))
# #                     eps = privacy_engine.accountant.get_epsilon(delta)
# #                     logging.info(f"[DP] ε≈{eps:.2f}, δ={delta}, "
# #                                  f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm',1.0)}")
# #                 return loss.item()

# #     if reg is not None:
# #         logging.info('loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
# #     else:
# #         logging.info('loss: {}'.format(loss.item()))

# #     if privacy_engine is not None:
# #         delta = float(dp.get("delta", 1e-5))
# #         eps = privacy_engine.accountant.get_epsilon(delta)
# #         logging.info(f"[DP] ε≈{eps:.2f}, δ={delta}, "
# #                      f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm',1.0)}")

# #     return loss.item()



# def train(model, trainloader, optimizer, epochs, reg=None, dp: Optional[dict]=None):
#     model.to(device)
#     model.train()
#     criterion = nn.CrossEntropyLoss().to(device)

#     # --- Enable DP-SGD (Opacus) if requested ---
#     privacy_engine = None
#     if dp and dp.get("enable", False):
#         try:
#             from opacus import PrivacyEngine
#             noise_multiplier = float(dp.get("noise_multiplier", 0.8))
#             max_grad_norm    = float(dp.get("max_grad_norm", 1.0))
#             secure_mode      = bool(dp.get("secure_mode", False))
#             accountant       = dp.get("accountant", "rdp")  # "rdp" works well

#             privacy_engine = PrivacyEngine(accountant=accountant, secure_mode=secure_mode)
#             model, optimizer, trainloader = privacy_engine.make_private(
#                 module=model,
#                 optimizer=optimizer,
#                 data_loader=trainloader,
#                 noise_multiplier=noise_multiplier,
#                 max_grad_norm=max_grad_norm,
#             )
#             logging.info(f"[DP] Enabled: sigma={noise_multiplier}, C={max_grad_norm}, accountant={accountant}")
#         except Exception as e:
#             logging.warning(f"[DP] Failed to enable DP-SGD ({e}). Continuing without DP.")
#             privacy_engine = None

#     # --- Regularization snapshot (unchanged) ---
#     if reg is not None:
#         old_weights = flatten_weights(extract_weights(model))
#         old_weights = torch.from_numpy(old_weights)

#     # --- Training loop ---
#     for epoch in range(1, epochs + 1):
#         for batch_id, (image, label) in enumerate(trainloader):
#             image, label = image.to(device), label.to(device)
#             optimizer.zero_grad()
#             output = model(image)
#             loss = criterion(output, label)

#             if reg is not None:
#                 new_weights = flatten_weights(extract_weights(model))
#                 new_weights = torch.from_numpy(new_weights)
#                 mse_loss = nn.MSELoss(reduction='sum')
#                 l2_loss = rou/2 * mse_loss(new_weights, old_weights)
#                 l2_loss = l2_loss.to(torch.float32)
#                 loss += l2_loss

#             loss.backward()
#             optimizer.step()

#             if batch_id % log_interval == 0:
#                 logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(epoch, epochs, loss.item()))

#             # Early stop
#             if loss.item() < loss_thres:
#                 # DP status when exiting early
#                 if privacy_engine is not None:
#                     try:
#                         delta = float(dp.get("delta", 1e-5))
#                         # Opacus API compatibility:
#                         eps = (privacy_engine.accountant.get_epsilon(delta)
#                                if hasattr(privacy_engine, "accountant")
#                                else privacy_engine.get_epsilon(delta))
#                         logging.info(f"[DP] ε≈{eps:.2f}, δ={delta}, "
#                                      f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm',1.0)}")
#                     except Exception as e:
#                         logging.debug(f"[DP] epsilon unavailable: {e}")
#                 return loss.item()

#         # Per-epoch DP logging
#         if privacy_engine is not None:
#             try:
#                 delta = float(dp.get("delta", 1e-5))
#                 eps = (privacy_engine.accountant.get_epsilon(delta)
#                        if hasattr(privacy_engine, "accountant")
#                        else privacy_engine.get_epsilon(delta))
#                 logging.info(f"[DP] ε≈{eps:.2f}, δ={delta}, "
#                              f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm',1.0)}")
#             except Exception as e:
#                 logging.debug(f"[DP] epsilon unavailable: {e}")

#     # Final loss logging
#     if reg is not None:
#         logging.info('loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
#     else:
#         logging.info('loss: {}'.format(loss.item()))

#     # Final DP logging
#     if privacy_engine is not None:
#         try:
#             delta = float(dp.get("delta", 1e-5))
#             eps = (privacy_engine.accountant.get_epsilon(delta)
#                    if hasattr(privacy_engine, "accountant")
#                    else privacy_engine.get_epsilon(delta))
#             logging.info(f"[DP] ε≈{eps:.2f}, δ={delta}, "
#                          f"σ={dp.get('noise_multiplier', 0.8)}, C={dp.get('max_grad_norm',1.0)}")
#         except Exception as e:
#             logging.debug(f"[DP] epsilon unavailable: {e}")

#     return loss.item()



# # def train(model, trainloader, optimizer, epochs, reg=None):
# #     model.to(device)
# #     model.train()
# #     criterion = nn.CrossEntropyLoss().to(device)

# #     # Get the snapshot of weights when training starts, if regularization is on
# #     if reg is not None:
# #         old_weights = flatten_weights(extract_weights(model))
# #         old_weights = torch.from_numpy(old_weights)

# #     for epoch in range(1, epochs + 1):
# #         for batch_id, (image, label) in enumerate(trainloader):
# #             image, label = image.to(device), label.to(device)
# #             optimizer.zero_grad()
# #             output = model(image)
# #             loss = criterion(output, label)

# #             # Add regularization
# #             if reg is not None:
# #                 new_weights = flatten_weights(extract_weights(model))
# #                 new_weights = torch.from_numpy(new_weights)
# #                 mse_loss = nn.MSELoss(reduction='sum')
# #                 l2_loss = rou/2 * mse_loss(new_weights, old_weights)
# #                 l2_loss = l2_loss.to(torch.float32)
# #                 loss += l2_loss

# #             loss.backward()
# #             optimizer.step()
# #             if batch_id % log_interval == 0:
# #                 logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
# #                     epoch, epochs, loss.item()))

# #             # Stop training if model is already in good shape
# #             if loss.item() < loss_thres:
# #                 return loss.item()

# #     if reg is not None:
# #         logging.info(
# #             'loss: {} l2_loss: {}'.format(loss.item(), l2_loss.item()))
# #     else:
# #         logging.info(
# #             'loss: {}'.format(loss.item()))
# #     return loss.item()


# def test(model, testloader):
#     model.to(device)
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = len(testloader.dataset)
#     with torch.no_grad():
#         for image, label in testloader:
#             image, label = image.to(device), label.to(device)
#             output = model(image)
#             # sum up batch loss
#             test_loss += F.nll_loss(output, label, reduction='sum').item()
#             # get the index of the max log-probability
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(label.view_as(pred)).sum().item()

#     accuracy = correct / total
#     logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

#     return accuracy
