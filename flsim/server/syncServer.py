import logging
import pickle
import random
import math
import time
from threading import Thread
from server import Server
from network import Network
from .record import Record, Profile
from ctypes import *


class Group(object):
    """Basic async group."""

    def __init__(self, client_list):
        self.clients = client_list

    def set_download_time(self, download_time):
        self.download_time = download_time

    def set_aggregate_time(self):
        """Only run after client configuration"""
        assert (len(self.clients) > 0), "Empty clients in group init!"
        self.delay = max([c.delay for c in self.clients])
        self.aggregate_time = self.download_time + self.delay

        # Get average throughput contributed by this group
        assert (self.clients[0].model_size > 0), "Zero model size in group init!"
        self.throughput = len(self.clients) * self.clients[0].model_size / \
                          self.aggregate_time

    def __eq__(self, other):
        return self.aggregate_time == other.aggregate_time

    def __ne__(self, other):
        return self.aggregate_time != other.aggregate_time

    def __lt__(self, other):
        return self.aggregate_time < other.aggregate_time

    def __le__(self, other):
        return self.aggregate_time <= other.aggregate_time

    def __gt__(self, other):
        return self.aggregate_time > other.aggregate_time

    def __ge__(self, other):
        return self.aggregate_time >= other.aggregate_time


class SyncServer(Server):
    """Synchronous federated learning server."""

    def make_clients(self, num_clients):
        super().make_clients(num_clients)

        # Set link speed for clients
        speed = []
        for client in self.clients:
            client.set_link(self.config)
            speed.append(client.speed_mean)

        logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

        # Initiate client profile of loss and delay
        self.profile = Profile(num_clients)
        if self.config.data.IID == False:
            print("Non IID")
            self.profile.set_primary_label([client.pref for client in self.clients])

    # Run synchronous federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        network = Network(self.config)  # create ns3 network/start ns3 program
        # dummy call to access

        # Init self accuracy records
        self.records = Record()

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        time.sleep(1)
        network.connect()
        # Perform rounds of federated learning
        T_old = 0.0
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the sync federated learning round
            accuracy, T_new = self.sync_round(round, T_old, network)
            logging.info('Round finished at time {} s\n'.format(T_new))

            # Update time
            T_old = T_new

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

        network.disconnect()

    def sync_round(self, round, T_old, network):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        sample_groups = self.selection(network)
        sample_clients, throughput = [], []
        delays = []
        dropouts = 0
        for group in sample_groups:
            parsed_clients = network.parse_clients(group.clients)
            simdata = network.sendRequest(requestType=1, array=parsed_clients)
            for client in group.clients:
                if simdata[client.client_id]["roundTime"] < 0:
                    client.delay = 0
                    dropouts = dropouts + 1
                    print("skip " + str(client.client_id))
                    print("roundTime" + str(simdata[client.client_id]["roundTime"]))
                    continue
                client.est_delay = simdata[client.client_id]["roundTime"]
                client.delay = simdata[client.client_id]["roundTime"]
                delays.append(simdata[client.client_id]["roundTime"])
                sample_clients.append(client)
                throughput.append(simdata[client.client_id]["throughput"])
            group.set_download_time(T_old)
            group.set_aggregate_time()
        self.throughput = 0
        if len(throughput) > 0:
            self.throughput = sum([t for t in throughput]) / len(throughput)
        print("throughputs")
        print(throughput)
        print("dropouts: " + str(dropouts))

        logging.info('Avg throughput {} kB/s'.format(self.throughput))

        # Configure sample clients
        self.configuration(sample_clients)

        # Use the max delay in all sample clients as the delay in sync round
        # delays = network.access_network(sample_clients)
        print(delays)
        max_delay = max(delays)  # access latency from ns3 simulation
        print(max_delay)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        T_cur = T_old + max_delay  # Update current time

        # Receive client updates
        reports = self.reporting(sample_clients)

        # Update profile and plot
        self.update_profile(reports)
        # Plot every plot_interval
        if math.floor(T_cur / self.config.plot_interval) > \
                math.floor(T_old / self.config.plot_interval):
            self.profile.plot(T_cur, self.config.paths.plot)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%'.format(100 * accuracy))
        self.records.append_record(T_cur, accuracy, self.throughput, dropouts, round)
        return self.records.get_latest_acc(), self.records.get_latest_t()

    def selection(self, network):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round
        select_type = self.config.clients.selection

        if select_type == 'random':
        # Select clients randomly
            sample_clients = [client for client in random.sample(
                self.clients, clients_per_round)]
            print("random")

        elif select_type == 'short_latency_first':
            # Select the clients with short latencies and random loss
            sample_clients = sorted(self.clients, key=lambda c:c.est_delay)
            sample_clients = sample_clients[:clients_per_round]
            print(select_type)

        elif select_type == 'high_loss_first':
            # Select the clients with random latencies and high loss
            sample_clients = sorted(self.clients, key=lambda c:c.loss, reverse = True)
            sample_clients = sample_clients[:clients_per_round]
            print(select_type)

        elif select_type == 'short_latency_high_loss_first':
            # Get the non-negative losses and delays
            losses = [c.loss for c in self.clients]
            losses_norm = [l/max(losses) for l in losses]
            delays = [c.est_delay for c in self.clients]
            delays_norm = [d/max(losses) for d in delays]

            # Sort the clients by jointly consider latency and loss
            gamma = 0.2
            #0.2 for mnist
            sorted_idx = sorted(range(len(self.clients)),
                                key=lambda i: losses_norm[i] - gamma * delays_norm[i],
                                reverse=True)
            print([losses[i] for i in sorted_idx])
            print([delays[i] for i in sorted_idx])
            sample_clients = [self.clients[i] for i in sorted_idx]
            sample_clients = sample_clients[:clients_per_round]
            print(select_type)
        # In sync case, create one group of all selected clients
        sample_groups = [Group([client for client in sample_clients])]

        return sample_groups

    def update_profile(self, reports):
        for report in reports:
            self.profile.update(report.client_id, report.loss, report.delay,
                                self.flatten_weights(report.weights))
