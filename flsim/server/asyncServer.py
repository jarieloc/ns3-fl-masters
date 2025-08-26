# flsim/server/asyncServer.py
import logging
import pickle
import random
import os
import time
import torch

from server import Server
from network import Network
from .record import Record, Profile


def _get(root, path, default=None):
    cur = root
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif hasattr(cur, k):
            cur = getattr(cur, k)
        else:
            return default
    return cur


class AsyncServer(Server):
    """Asynchronous federated learning server."""

    def readAsyncResponse(self):
        """
        Poll once:
        - returns {} while ns-3 is running
        - then returns one client’s dict at a time: {id: {...}}
        - when all delivered, returns 'end'
        """
        # nothing ever started
        if self._proc is None and not getattr(self, "_async_queue", None):
            return 'end'

        # still running -> no results yet
        if self._proc is not None and self._proc.poll() is None:
            return {}

        # finished: if queue not built yet, parse and build single-client chunks
        if self._proc is not None:
            stdout = self._proc.stdout.read() if self._proc.stdout else ''
            _ = self._proc.stderr.read() if self._proc.stderr else ''
            self._proc = None  # free the handle

            data = self._parse_last_json(stdout)  # your existing helper
            # map local -> real ids (populated in sendAsyncRequest)
            id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
            results = {int(e['id']): e for e in data.get('clientResults', [])}

            # init queue if needed
            if not hasattr(self, "_async_queue"):
                self._async_queue = []

            # build per-client messages
            for local in range(len(self._async_ids)):
                ent = results.get(local)
                if not ent:
                    continue
                real_id = id_map[local]
                rx_bytes = float(ent.get('rxBytes', 0.0))
                done_at  = float(ent.get('doneAt', -1.0))
                thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
                self._async_queue.append({
                    real_id: {
                        'startTime': 0.0,
                        'endTime': done_at if done_at >= 0 else self._thz_cfg['sim_time'],
                        'throughput': thr,
                    }
                })

        # deliver one dict per call until queue is empty
        if getattr(self, "_async_queue", None):
            return self._async_queue.pop(0)

        return 'end'


    # ---------- small helpers ----------
    def _get_dp_cfg(self):
        """
        Find DP config no matter where it lives:
        - config.dp
        - config.fl.dp
        - config.federated_learning.dp
        - dict forms if the loader kept raw JSON
        """
        for paths in (
            [['dp']],
            [['fl', 'dp'], ['federated_learning', 'dp']],
        ):
            for p in paths:
                val = _get(self.config, p, None)
                if val:
                    return val

        # raw dict fallback
        raw = getattr(self.config, '_raw', None) or getattr(self.config, 'raw', None)
        if isinstance(raw, dict):
            for p in (['dp'], ['fl', 'dp'], ['federated_learning', 'dp']):
                cur = raw
                ok = True
                for k in p:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok:
                    return cur
        return None

    def _handle_one_result(self, client_id, metrics, id_to_client, throughputs):
        if client_id == -1 or client_id not in id_to_client:
            return None
        select_client, T_client = id_to_client[client_id]
        # prefer endTime (async) but accept roundTime (sync)
        delay = float(metrics.get("endTime", metrics.get("roundTime", 0.0)) or 0.0)
        select_client.delay = delay
        throughputs.append(float(metrics.get("throughput", 0.0)))
        return select_client, T_client

    # ---------- lifecycle ----------
    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model
        logging.info('Model: {}'.format(model_type))

        self.model = fl_model.Net()
        self.async_save_model(self.model, model_path, 0.0)

        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # initial snapshot

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
        if _get(self.config, ['data', 'IID'], True) is False:
            self.profile.set_primary_label([client.pref for client in self.clients])

    def run(self):
        self.records = Record()
        self.throughput = 0.0  # kB/s; avoid AttributeError if a round yields no results

        rounds = _get(self.config, ['fl', 'rounds'], _get(self.config, ['federated_learning', 'rounds'], 1))
        target_accuracy = _get(self.config, ['fl', 'target_accuracy'],
                               _get(self.config, ['federated_learning', 'target_accuracy'], None))
        reports_path = self.config.paths.reports

        # Init async/staleness parameters (support "sync" or "async" naming)
        sync_cfg = _get(self.config, ['sync'], None) or _get(self.config, ['async'], None)
        self.alpha = getattr(sync_cfg, 'alpha', _get(sync_cfg, ['alpha'], 0.9))
        self.staleness_func = getattr(sync_cfg, 'staleness_func', _get(sync_cfg, ['staleness_func'], 'polynomial'))

        # Resolve DP config once at startup
        self._dp_cfg = self._get_dp_cfg()

        network = Network(self.config)
        logging.info(f"[DP] server-level cfg: {self._dp_cfg}")

        self.records = Record()

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        T_old = 0.0
        time.sleep(0.5)
        network.connect()

        f = open("dropout.txt", "a")
        try:
            for rnd in range(1, rounds + 1):
                logging.info('**** Round {}/{} ****'.format(rnd, rounds))
                f.write('**** Round {}/{} ****\n'.format(rnd, rounds)); f.flush()

                self.rm_old_models(self.config.paths.model, T_old)
                accuracy, T_new = self.async_round(rnd, T_old, network, f)
                T_old = T_new

                if target_accuracy and (accuracy >= target_accuracy):
                    logging.info('Target accuracy reached.')
                    break

            if reports_path:
                with open(reports_path, 'wb') as f_out:
                    pickle.dump(self.saved_reports, f_out)
                logging.info('Saved reports: {}'.format(reports_path))
        finally:
            network.disconnect()
            f.close()

    def async_round(self, round_idx, T_old, network, f):
        import fl_model  # pylint: disable=import-error
        target_accuracy = _get(self.config, ['fl', 'target_accuracy'],
                               _get(self.config, ['federated_learning', 'target_accuracy'], None))

        # Select clients
        sample_clients = self.selection()
        parsed_clients = network.parse_clients(sample_clients)

        id_to_client = {c.client_id: (c, T_old) for c in sample_clients}
        client_finished = {c.client_id: False for c in sample_clients}

        # Try true-async; if not implemented, fall back to a one-shot sync run
        # THz backend is verbose; avoid async Popen deadlocks: force sync path for THz
        use_async = not getattr(network, "_use_thz", False)
        if use_async:
            try:
                network.sendAsyncRequest(requestType=1, array=parsed_clients)
            except NotImplementedError:
                use_async = False


        T_new = T_old
        throughputs = []
        self.throughput = 0.0  # reset per-round; will be updated when results arrive


        def _apply_update(select_client, T_client):
            nonlocal T_new
            self.async_configuration([select_client], T_client)
            select_client.run(reg=True)
            T_cur = T_client + select_client.delay
            T_new = T_cur

            logging.info('Training finished on clients {} at time {} s'.format(select_client, T_cur))

            reports = self.reporting([select_client])

            self.update_profile(reports)
            logging.info('Aggregating updates from clients {}'.format(select_client))
            staleness = select_client.delay
            updated_weights = self.aggregation(reports, staleness)

            fl_model.load_weights(self.model, updated_weights)

            if self.config.paths.reports:
                self.save_reports(round_idx, reports)
            self.async_save_model(self.model, self.config.paths.model, T_cur)

            if _get(self.config, ['clients', 'do_test'], False):
                acc = self.accuracy_averaging(reports)
            else:
                testset = self.loader.get_testset()
                batch_size = _get(self.config, ['fl', 'batch_size'],
                                  _get(self.config, ['federated_learning', 'batch_size'], 32))
                testloader = fl_model.get_testloader(testset, batch_size)
                acc = fl_model.test(self.model, testloader)

            # throughput accounting (avg of delivered so far)
            # self.throughput = (sum(throughputs) / len(throughputs)) if throughputs else 0.0
            self.throughput = ((sum(throughputs) / len(throughputs)) / 1024.0) if throughputs else 0.0
            logging.info('Average accuracy: {:.2f}%\n'.format(100 * acc))
            self.records.async_time_graphs(T_cur, acc, self.throughput)
            return acc

        # ---------- SYNC FALLBACK ----------
        if not use_async:
            data = network.sendRequest(requestType=1, array=parsed_clients) or {}
            ordered = sorted(
                data.items(),
                key=lambda kv: kv[1].get('endTime', kv[1].get('roundTime', float('inf')))
            )
            for cid, metrics in ordered:
                handled = self._handle_one_result(cid, metrics, id_to_client, throughputs)
                if not handled:
                    continue
                select_client, T_client = handled
                client_finished[cid] = True
                acc = _apply_update(select_client, T_client)
                if target_accuracy and (self.records.get_latest_acc() >= target_accuracy):
                    logging.info('Target accuracy reached.')
                    break

            # If nothing was handled (e.g., sim returned empty), record a no-op to avoid IndexError
            if not any(client_finished.values()):
                self.records.async_time_graphs(T_old, 0.0, 0.0)

            logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(T_new, self.throughput))
            # count unfinished
            cnt = sum(0 if client_finished[c] else 1 for c in client_finished)
            for c in client_finished:
                if not client_finished[c]:
                    f.write(str(c) + '\n'); f.flush()
            self.records.async_round_graphs(round_idx, cnt)
            return self.records.get_latest_acc(), self.records.get_latest_t()

        # ---------- TRUE ASYNC ----------
        while True:
            simdata = network.readAsyncResponse()
            if simdata == 'end':
                break
            if not simdata:
                continue  # nothing this tick

            items = simdata.items() if isinstance(simdata, dict) else []
            for cid, metrics in items:
                handled = self._handle_one_result(cid, metrics, id_to_client, throughputs)
                if not handled:
                    continue
                select_client, T_client = handled
                client_finished[cid] = True
                acc = _apply_update(select_client, T_client)
                if target_accuracy and (self.records.get_latest_acc() >= target_accuracy):
                    logging.info('Target accuracy reached.')
                    break

        logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(T_new, self.throughput))
        cnt = sum(0 if client_finished[c] else 1 for c in client_finished)
        for c in client_finished:
            if not client_finished[c]:
                f.write(str(c) + '\n'); f.flush()
        self.records.async_round_graphs(round_idx, cnt)
        return self.records.get_latest_acc(), self.records.get_latest_t()

    # ---------- selection / configuration ----------
    def selection(self):
        clients_per_round = self.config.clients.per_round
        select_type = self.config.clients.selection

        if select_type == 'random':
            sample_clients = [client for client in random.sample(self.clients, clients_per_round)]

        elif select_type == 'short_latency_first':
            sample_clients = sorted(self.clients, key=lambda c: c.est_delay)[:clients_per_round]
            print(sample_clients)

        elif select_type == 'short_latency_high_loss_first':
            losses = [c.loss for c in self.clients]
            losses_norm = [l / max(losses) for l in losses]
            delays = [c.est_delay for c in self.clients]
            delays_norm = [d / max(losses) for d in delays]
            gamma = 0.2
            sorted_idx = sorted(
                range(len(self.clients)),
                key=lambda i: losses_norm[i] - gamma * delays_norm[i],
                reverse=True
            )
            print([losses[i] for i in sorted_idx])
            print([delays[i] for i in sorted_idx])
            sample_clients = [self.clients[i] for i in sorted_idx][:clients_per_round]
            print(sample_clients)
        else:
            sample_clients = [client for client in random.sample(self.clients, clients_per_round)]
        return sample_clients

    def async_configuration(self, sample_clients, download_time):
        loader_type = _get(self.config, ['loader'], None)
        loading = _get(self.config, ['data', 'loading'], 'dynamic')

        if loading == 'dynamic' and loader_type == 'shard':
            self.loader.create_shards()

        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)

            # Pass the already-resolved DP config to each client
            client.dp = self._dp_cfg
            logging.info(f"[DP] cfg for client {client.client_id}: {client.dp}")

            # Continue configuration on client
            client.async_configure(self.config, download_time)

    # ---------- aggregation / staleness ----------
    def aggregation(self, reports, staleness=None):
        return self.federated_async(reports, staleness)

    def extract_client_weights(self, reports):
        return [report.weights for report in reports]

    def federated_async(self, reports, staleness):
        import fl_model  # pylint: disable=import-error

        weights = self.extract_client_weights(reports)
        total_samples = sum([report.num_samples for report in reports])

        new_weights = [torch.zeros(x.size()) for _, x in weights[0]]
        for i, update in enumerate(weights):
            num_samples = reports[i].num_samples
            for j, (_, weight) in enumerate(update):
                new_weights[j] += weight * (num_samples / total_samples)

        baseline_weights = fl_model.extract_weights(self.model)

        alpha_t = self.alpha * self.staleness(staleness)
        logging.info('{} staleness: {} alpha_t: {}'.format(self.staleness_func, staleness, alpha_t))

        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, (1 - alpha_t) * weight + alpha_t * new_weights[i]))
        return updated_weights

    def staleness(self, staleness):
        if self.staleness_func == "constant":
            return 1
        elif self.staleness_func == "polynomial":
            a = 0.5
            return pow(staleness + 1, -a)
        elif self.staleness_func == "hinge":
            a, b = 10, 4
            if staleness <= b:
                return 1
            else:
                return 1 / (a * (staleness - b) + 1)

    # ---------- model snapshots / housekeeping ----------
    def async_save_model(self, model, path, download_time):
        path += '/global_' + '{}'.format(download_time)
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def rm_old_models(self, path, cur_time):
        for filename in os.listdir(path):
            try:
                model_time = float(filename.split('_')[1])
                if model_time < cur_time:
                    os.remove(os.path.join(path, filename))
                    logging.info('Remove model {}'.format(filename))
            except Exception as e:
                logging.debug(e)
                continue

    def update_profile(self, reports):
        for report in reports:
            self.profile.update(report.client_id, report.loss, report.delay,
                                self.flatten_weights(report.weights))


# # flsim/server/asyncServer.py
# import logging
# import pickle
# import random
# import math
# from threading import Thread
# import torch
# import time
# from queue import PriorityQueue
# import os

# from server import Server
# from network import Network
# from .record import Record, Profile


# class AsyncServer(Server):
#     """Asynchronous federated learning server."""

#     # ---------- small helpers ----------
#     def _get_dp_cfg(self):
#         """
#         Try to find a DP config regardless of where it was placed in the JSON.
#         Looks in: config.dp, config.fl.dp, config.federated_learning.dp,
#         or raw dict copies if the loader stashed them.
#         """
#         cfg = getattr(self.config, 'dp', None)
#         if cfg:
#             return cfg

#         fl = getattr(self.config, 'fl', None) or getattr(self.config, 'federated_learning', None)
#         if fl:
#             cfg = getattr(fl, 'dp', None)
#             if cfg:
#                 return cfg

#         raw = getattr(self.config, '_raw', None) or getattr(self.config, 'raw', None)
#         if isinstance(raw, dict):
#             return raw.get('dp') or (raw.get('federated_learning', {}) or {}).get('dp')

#         return None

#     def _handle_one_result(self, client_id, metrics, id_to_client, throughputs):
#         """
#         Map a (client_id, metrics) pair coming from Network to a local client.
#         Returns (select_client, T_client) or None if the id is invalid/unknown.
#         """
#         if client_id == -1 or client_id not in id_to_client:
#             # Ignore sentinel / unknown IDs quietly
#             return None

#         select_client, T_client = id_to_client[client_id]
#         # prefer endTime (async) but accept roundTime (sync)
#         delay = float(metrics.get("endTime", metrics.get("roundTime", 0.0)) or 0.0)
#         select_client.delay = delay
#         throughputs.append(float(metrics.get("throughput", 0.0)))
#         return select_client, T_client

#     # ---------- lifecycle ----------
#     def load_model(self):
#         import fl_model  # pylint: disable=import-error

#         model_path = self.config.paths.model
#         model_type = self.config.model

#         logging.info('Model: {}'.format(model_type))

#         # Set up global model
#         self.model = fl_model.Net()
#         self.async_save_model(self.model, model_path, 0.0)

#         # Extract flattened weights (if applicable)
#         if self.config.paths.reports:
#             self.saved_reports = {}
#             self.save_reports(0, [])  # Save initial model

#     def make_clients(self, num_clients):
#         super().make_clients(num_clients)

#         # Set link speed for clients
#         speed = []
#         for client in self.clients:
#             client.set_link(self.config)
#             speed.append(client.speed_mean)

#         logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

#         # Initiate client profile of loss and delay
#         self.profile = Profile(num_clients)
#         if self.config.data.IID == False:
#             self.profile.set_primary_label([client.pref for client in self.clients])

#     # Run asynchronous federated learning
#     def run(self):
#         rounds = self.config.fl.rounds
#         target_accuracy = self.config.fl.target_accuracy
#         reports_path = self.config.paths.reports

#         # Init async/staleness parameters (handle either "sync" or "async" block in JSON)
#         sync_cfg = getattr(self.config, 'sync', None) or getattr(self.config, 'async', None)
#         self.alpha = sync_cfg.alpha
#         self.staleness_func = sync_cfg.staleness_func

#         # Resolve DP config once at startup
#         self._dp_cfg = self._get_dp_cfg()

#         network = Network(self.config)  # create ns3 network/start ns3 program
#         logging.info(f"[DP] server-level cfg: {self._dp_cfg}")

#         # Init self accuracy records
#         self.records = Record()

#         if target_accuracy:
#             logging.info('Training: {} rounds or {}% accuracy\n'.format(
#                 rounds, 100 * target_accuracy))
#         else:
#             logging.info('Training: {} rounds\n'.format(rounds))

#         # Perform rounds of federated learning
#         T_old = 0.0

#         # connect to network
#         time.sleep(1)
#         network.connect()

#         f = open("dropout.txt", "a")

#         for round in range(1, rounds + 1):
#             logging.info('**** Round {}/{} ****'.format(round, rounds))
#             f.write('**** Round {}/{} ****'.format(round, rounds))
#             f.write('\n'); f.flush()

#             # Perform async rounds with current grouping strategy
#             self.rm_old_models(self.config.paths.model, T_old)
#             accuracy, T_new = self.async_round(round, T_old, network, f)

#             # Update time
#             T_old = T_new

#             # Break when target accuracy is met
#             if target_accuracy and (accuracy >= target_accuracy):
#                 logging.info('Target accuracy reached.')
#                 break

#         if reports_path:
#             with open(reports_path, 'wb') as f_out:
#                 pickle.dump(self.saved_reports, f_out)
#             logging.info('Saved reports: {}'.format(reports_path))

#         network.disconnect()
#         f.close()

#     def async_round(self, round, T_old, network, f):
#         """Run one async round (tolerates sync or async Network backends)."""
#         import fl_model  # pylint: disable=import-error
#         target_accuracy = self.config.fl.target_accuracy

#         # Pick participating clients
#         sample_clients = self.selection()
#         parsed_clients = network.parse_clients(sample_clients)

#         # Map ids -> (client, T_old); track who finished
#         id_to_client = {c.client_id: (c, T_old) for c in sample_clients}
#         client_finished = {c.client_id: False for c in sample_clients}

#         # Try true-async; if not implemented, fall back to a one-shot sync run
#         use_async = True
#         try:
#             network.sendAsyncRequest(requestType=1, array=parsed_clients)
#         except NotImplementedError:
#             use_async = False

#         T_new = T_old
#         throughputs = []

#         def _apply_update(select_client, T_client):
#             nonlocal T_new
#             # configure + train
#             self.async_configuration([select_client], T_client)
#             select_client.run(reg=True)
#             T_cur = T_client + select_client.delay
#             T_new = T_cur

#             logging.info('Training finished on clients {} at time {} s'.format(
#                 select_client, T_cur
#             ))

#             # receive report
#             reports = self.reporting([select_client])

#             # update profile + aggregate
#             self.update_profile(reports)
#             logging.info('Aggregating updates from clients {}'.format(select_client))
#             staleness = select_client.delay
#             updated_weights = self.aggregation(reports, staleness)

#             # load new global
#             fl_model.load_weights(self.model, updated_weights)

#             # save reports / model snapshot
#             if self.config.paths.reports:
#                 self.save_reports(round, reports)
#             self.async_save_model(self.model, self.config.paths.model, T_cur)

#             # accuracy
#             if self.config.clients.do_test:
#                 acc = self.accuracy_averaging(reports)
#             else:
#                 testset = self.loader.get_testset()
#                 batch_size = self.config.fl.batch_size
#                 testloader = fl_model.get_testloader(testset, batch_size)
#                 acc = fl_model.test(self.model, testloader)

#             self.throughput = (sum(throughputs) / len(throughputs)) if throughputs else 0.0
#             logging.info('Average accuracy: {:.2f}%\n'.format(100 * acc))
#             self.records.async_time_graphs(T_cur, acc, self.throughput)

#             return acc

#         # === SYNC FALLBACK: one shot result dict ===
#         if not use_async:
#             data = network.sendRequest(requestType=1, array=parsed_clients) or {}
#             # process in the order of finish time if available
#             ordered = sorted(
#                 data.items(),
#                 key=lambda kv: kv[1].get('endTime', kv[1].get('roundTime', float('inf')))
#             )
#             for cid, metrics in ordered:
#                 handled = self._handle_one_result(cid, metrics, id_to_client, throughputs)
#                 if not handled:
#                     continue
#                 select_client, T_client = handled
#                 client_finished[cid] = True
#                 acc = _apply_update(select_client, T_client)
#                 if target_accuracy and (self.records.get_latest_acc() >= target_accuracy):
#                     logging.info('Target accuracy reached.')
#                     break

#             # bookkeeping + plots
#             logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(
#                 T_new, self.throughput
#             ))
#             cnt = sum(0 if client_finished[c] else 1 for c in client_finished)
#             for c in client_finished:
#                 if not client_finished[c]:
#                     f.write(str(c) + '\n'); f.flush()
#             self.records.async_round_graphs(round, cnt)
#             return self.records.get_latest_acc(), self.records.get_latest_t()

#         # === TRUE ASYNC: poll until 'end' ===
#         while True:
#             simdata = network.readAsyncResponse()
#             if simdata == 'end':
#                 break
#             if not simdata:
#                 continue  # nothing this tick

#             # Some backends stream a single id; others batch several at once.
#             items = simdata.items() if isinstance(simdata, dict) else []
#             for cid, metrics in items:
#                 handled = self._handle_one_result(cid, metrics, id_to_client, throughputs)
#                 if not handled:
#                     continue
#                 select_client, T_client = handled
#                 client_finished[cid] = True
#                 acc = _apply_update(select_client, T_client)
#                 if target_accuracy and (self.records.get_latest_acc() >= target_accuracy):
#                     logging.info('Target accuracy reached.')
#                     break

#         # wrap-up
#         logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(
#             T_new, self.throughput
#         ))
#         cnt = sum(0 if client_finished[c] else 1 for c in client_finished)
#         for c in client_finished:
#             if not client_finished[c]:
#                 f.write(str(c) + '\n'); f.flush()
#         self.records.async_round_graphs(round, cnt)
#         return self.records.get_latest_acc(), self.records.get_latest_t()

#     # ---------- selection / configuration ----------
#     def selection(self):
#         # Select devices to participate in round
#         clients_per_round = self.config.clients.per_round
#         select_type = self.config.clients.selection

#         if select_type == 'random':
#             # Select clients randomly
#             sample_clients = [client for client in random.sample(
#                 self.clients, clients_per_round)]

#         elif select_type == 'short_latency_first':
#             # Select the clients with short latencies and random loss
#             sample_clients = sorted(self.clients, key=lambda c: c.est_delay)
#             sample_clients = sample_clients[:clients_per_round]
#             print(sample_clients)

#         elif select_type == 'short_latency_high_loss_first':
#             # Get the non-negative losses and delays
#             losses = [c.loss for c in self.clients]
#             losses_norm = [l / max(losses) for l in losses]
#             delays = [c.est_delay for c in self.clients]
#             delays_norm = [d / max(losses) for d in delays]

#             # Sort the clients by jointly consider latency and loss
#             gamma = 0.2
#             sorted_idx = sorted(
#                 range(len(self.clients)),
#                 key=lambda i: losses_norm[i] - gamma * delays_norm[i],
#                 reverse=True
#             )
#             print([losses[i] for i in sorted_idx])
#             print([delays[i] for i in sorted_idx])
#             sample_clients = [self.clients[i] for i in sorted_idx]
#             sample_clients = sample_clients[:clients_per_round]
#             print(sample_clients)

#         return sample_clients

#     def async_configuration(self, sample_clients, download_time):
#         loader_type = self.config.loader
#         loading = self.config.data.loading

#         if loading == 'dynamic':
#             # Create shards if applicable
#             if loader_type == 'shard':
#                 self.loader.create_shards()

#         # Configure selected clients for federated learning task
#         for client in sample_clients:
#             if loading == 'dynamic':
#                 self.set_client_data(client)  # Send data partition to client

#             # Pass the already-resolved DP config to each client
#             client.dp = self._dp_cfg
#             logging.info(f"[DP] cfg for client {client.client_id}: {client.dp}")

#             # Continue configuration on client (loads the snapshot, etc.)
#             client.async_configure(self.config, download_time)

#     # ---------- aggregation ----------
#     def aggregation(self, reports, staleness=None):
#         return self.federated_async(reports, staleness)

#     def extract_client_weights(self, reports):
#         # Extract weights from reports
#         weights = [report.weights for report in reports]
#         return weights

#     def federated_async(self, reports, staleness):
#         import fl_model  # pylint: disable=import-error

#         # Extract updates from reports
#         weights = self.extract_client_weights(reports)

#         # Extract total number of samples
#         total_samples = sum([report.num_samples for report in reports])

#         # Perform weighted averaging
#         new_weights = [torch.zeros(x.size())  # pylint: disable=no-member
#                        for _, x in weights[0]]
#         for i, update in enumerate(weights):
#             num_samples = reports[i].num_samples
#             for j, (_, weight) in enumerate(update):
#                 # Use weighted average by number of samples
#                 new_weights[j] += weight * (num_samples / total_samples)

#         # Extract baseline model weights - latest model
#         baseline_weights = fl_model.extract_weights(self.model)

#         # Calculate the staleness-aware weights
#         alpha_t = self.alpha * self.staleness(staleness)
#         logging.info('{} staleness: {} alpha_t: {}'.format(
#             self.staleness_func, staleness, alpha_t
#         ))

#         # Load updated weights into model
#         updated_weights = []
#         for i, (name, weight) in enumerate(baseline_weights):
#             updated_weights.append(
#                 (name, (1 - alpha_t) * weight + alpha_t * new_weights[i])
#             )

#         return updated_weights

#     def staleness(self, staleness):
#         if self.staleness_func == "constant":
#             return 1
#         elif self.staleness_func == "polynomial":
#             a = 0.5
#             return pow(staleness + 1, -a)
#         elif self.staleness_func == "hinge":
#             a, b = 10, 4
#             if staleness <= b:
#                 return 1
#             else:
#                 return 1 / (a * (staleness - b) + 1)

#     # ---------- model snapshots / housekeeping ----------
#     def async_save_model(self, model, path, download_time):
#         path += '/global_' + '{}'.format(download_time)
#         torch.save(model.state_dict(), path)
#         logging.info('Saved global model: {}'.format(path))

#     def rm_old_models(self, path, cur_time):
#         for filename in os.listdir(path):
#             try:
#                 model_time = float(filename.split('_')[1])
#                 if model_time < cur_time:
#                     os.remove(os.path.join(path, filename))
#                     logging.info('Remove model {}'.format(filename))
#             except Exception as e:
#                 logging.debug(e)
#                 continue

#     def update_profile(self, reports):
#         for report in reports:
#             self.profile.update(report.client_id, report.loss, report.delay,
#                                 self.flatten_weights(report.weights))



# import logging
# import pickle
# import random
# import math
# from threading import Thread
# import torch
# import time
# from queue import PriorityQueue
# import os
# from server import Server
# from network import Network
# from .record import Record, Profile


# class AsyncServer(Server):
#     """Asynchronous federated learning server."""

#     def _handle_one_result(self, client_id, metrics, id_to_client, throughputs):
#         """
#         Map a (client_id, metrics) pair coming from Network to a local client.
#         Returns (select_client, T_client) or None if the id is invalid/unknown.
#         """
#         if client_id == -1 or client_id not in id_to_client:
#             # Ignore sentinel / unknown IDs quietly
#             return None

#         select_client, T_client = id_to_client[client_id]
#         # prefer endTime (async) but accept roundTime (sync)
#         delay = float(metrics.get("endTime", metrics.get("roundTime", 0.0)) or 0.0)
#         select_client.delay = delay
#         throughputs.append(float(metrics.get("throughput", 0.0)))
#         return select_client, T_client


#     def load_model(self):
#         import fl_model  # pylint: disable=import-error

#         model_path = self.config.paths.model
#         model_type = self.config.model

#         logging.info('Model: {}'.format(model_type))

#         # Set up global model
#         self.model = fl_model.Net()
#         self.async_save_model(self.model, model_path, 0.0)

#         # Extract flattened weights (if applicable)
#         if self.config.paths.reports:
#             self.saved_reports = {}
#             self.save_reports(0, [])  # Save initial model

#     def make_clients(self, num_clients):
#         super().make_clients(num_clients)

#         # Set link speed for clients
#         speed = []
#         for client in self.clients:
#             client.set_link(self.config)
#             speed.append(client.speed_mean)

#         logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

#         # Initiate client profile of loss and delay
#         self.profile = Profile(num_clients)
#         if self.config.data.IID == False:
#             self.profile.set_primary_label([client.pref for client in self.clients])

#     # Run asynchronous federated learning
#     def run(self):
#         rounds = self.config.fl.rounds
#         target_accuracy = self.config.fl.target_accuracy
#         reports_path = self.config.paths.reports

#         # Init async parameters
#         self.alpha = self.config.sync.alpha
#         self.staleness_func = self.config.sync.staleness_func

#         network = Network(self.config)  # create ns3 network/start ns3 program
#         cfg_dp = getattr(self.config, 'dp', None) or getattr(getattr(self.config, 'fl', object()), 'dp', None)
#         logging.info(f"[DP] server-level cfg: {cfg_dp}")


#         # logging.info(f"[DP] server-level cfg: {getattr(self.config, 'dp', None)}")
#         # dummy call to access

#         # Init self accuracy records
#         self.records = Record()

#         if target_accuracy:
#             logging.info('Training: {} rounds or {}% accuracy\n'.format(
#                 rounds, 100 * target_accuracy))
#         else:
#             logging.info('Training: {} rounds\n'.format(rounds))

#         # Perform rounds of federated learning
#         T_old = 0.0

#         #connect to network
#         time.sleep(1)
#         network.connect()

#         f = open("dropout.txt", "a")

#         for round in range(1, rounds + 1):
#             logging.info('**** Round {}/{} ****'.format(round, rounds))
#             f.write('**** Round {}/{} ****'.format(round, rounds))
#             f.write('\n')
#             f.flush()
#             # Perform async rounds of federated learning with certain
#             # grouping strategy
#             self.rm_old_models(self.config.paths.model, T_old)
#             accuracy, T_new = self.async_round(round, T_old, network, f)

#             # Update time
#             T_old = T_new

#             # Break loop when target accuracy is met
#             if target_accuracy and (accuracy >= target_accuracy):
#                 logging.info('Target accuracy reached.')
#                 break

#         if reports_path:
#             with open(reports_path, 'wb') as f:
#                 pickle.dump(self.saved_reports, f)
#             logging.info('Saved reports: {}'.format(reports_path))

#         network.disconnect()
#         f.close()

#     def async_round(self, round, T_old, network, f):
#         """Run one async round (tolerates sync or async Network backends)."""
#         import fl_model  # pylint: disable=import-error
#         target_accuracy = self.config.fl.target_accuracy

#         # Pick participating clients
#         sample_clients = self.selection()
#         parsed_clients = network.parse_clients(sample_clients)

#         # Map ids -> (client, T_old); track who finished
#         id_to_client = {c.client_id: (c, T_old) for c in sample_clients}
#         client_finished = {c.client_id: False for c in sample_clients}

#         # Try true-async; if not implemented, fall back to a one-shot sync run
#         use_async = True
#         try:
#             network.sendAsyncRequest(requestType=1, array=parsed_clients)
#         except NotImplementedError:
#             use_async = False

#         T_new = T_old
#         throughputs = []

#         def _apply_update(select_client, T_client):
#             nonlocal T_new
#             # configure + train
#             self.async_configuration([select_client], T_client)
#             select_client.run(reg=True)
#             T_cur = T_client + select_client.delay
#             T_new = T_cur

#             logging.info('Training finished on clients {} at time {} s'.format(
#                 select_client, T_cur
#             ))

#             # receive report
#             reports = self.reporting([select_client])

#             # update profile + aggregate
#             self.update_profile(reports)
#             logging.info('Aggregating updates from clients {}'.format(select_client))
#             staleness = select_client.delay
#             updated_weights = self.aggregation(reports, staleness)

#             # load new global
#             fl_model.load_weights(self.model, updated_weights)

#             # save reports / model snapshot
#             if self.config.paths.reports:
#                 self.save_reports(round, reports)
#             self.async_save_model(self.model, self.config.paths.model, T_cur)

#             # accuracy
#             if self.config.clients.do_test:
#                 acc = self.accuracy_averaging(reports)
#             else:
#                 testset = self.loader.get_testset()
#                 batch_size = self.config.fl.batch_size
#                 testloader = fl_model.get_testloader(testset, batch_size)
#                 acc = fl_model.test(self.model, testloader)

#             self.throughput = (sum(throughputs) / len(throughputs)) if throughputs else 0.0
#             logging.info('Average accuracy: {:.2f}%\n'.format(100 * acc))
#             self.records.async_time_graphs(T_cur, acc, self.throughput)

#             return acc

#         # === SYNC FALLBACK: one shot result dict ===
#         if not use_async:
#             data = network.sendRequest(requestType=1, array=parsed_clients) or {}
#             # process in the order of finish time if available
#             ordered = sorted(
#                 data.items(),
#                 key=lambda kv: kv[1].get('endTime', kv[1].get('roundTime', float('inf')))
#             )
#             for cid, metrics in ordered:
#                 handled = self._handle_one_result(cid, metrics, id_to_client, throughputs)
#                 if not handled:
#                     continue
#                 select_client, T_client = handled
#                 client_finished[cid] = True
#                 acc = _apply_update(select_client, T_client)
#                 if target_accuracy and (self.records.get_latest_acc() >= target_accuracy):
#                     logging.info('Target accuracy reached.')
#                     break

#             # bookkeeping + plots
#             logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(
#                 T_new, self.throughput
#             ))
#             cnt = sum(0 if client_finished[c] else 1 for c in client_finished)
#             for c in client_finished:
#                 if not client_finished[c]:
#                     f.write(str(c) + '\n'); f.flush()
#             self.records.async_round_graphs(round, cnt)
#             return self.records.get_latest_acc(), self.records.get_latest_t()

#         # === TRUE ASYNC: poll until 'end' ===
#         while True:
#             simdata = network.readAsyncResponse()
#             if simdata == 'end':
#                 break
#             if not simdata:
#                 continue  # nothing this tick

#             # Some backends stream a single id; others batch several at once.
#             items = simdata.items() if isinstance(simdata, dict) else []
#             for cid, metrics in items:
#                 handled = self._handle_one_result(cid, metrics, id_to_client, throughputs)
#                 if not handled:
#                     continue
#                 select_client, T_client = handled
#                 client_finished[cid] = True
#                 acc = _apply_update(select_client, T_client)
#                 if target_accuracy and (self.records.get_latest_acc() >= target_accuracy):
#                     logging.info('Target accuracy reached.')
#                     break

#         # wrap-up
#         logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(
#             T_new, self.throughput
#         ))
#         cnt = sum(0 if client_finished[c] else 1 for c in client_finished)
#         for c in client_finished:
#             if not client_finished[c]:
#                 f.write(str(c) + '\n'); f.flush()
#         self.records.async_round_graphs(round, cnt)
#         return self.records.get_latest_acc(), self.records.get_latest_t()


#     # def async_round(self, round, T_old, network, f):
#     #     """Run one async round for T_async"""
#     #     import fl_model  # pylint: disable=import-error
#     #     target_accuracy = self.config.fl.target_accuracy

#     #     # Select clients to participate in the round
#     #     sample_clients = self.selection()

#     #     # Send selected clients to ns-3
#     #     parsed_clients = network.parse_clients(sample_clients)
#     #     network.sendAsyncRequest(requestType=1, array=parsed_clients)

#     #     id_to_client = {}
#     #     client_finished = {}
#     #     for client in sample_clients:
#     #         id_to_client[client.client_id] = (client, T_old)
#     #         client_finished[client.client_id] = False

#     #     T_new = T_old
#     #     throughputs = []
#     #     # Start the asynchronous updates
#     #     while True:
#     #         simdata = network.readAsyncResponse()

#     #         if simdata != 'end':
#     #             #get the client/group based on the id, use map
#     #             client_id = -1
#     #             for key in simdata:
#     #                 client_id = key

#     #             select_client = id_to_client[client_id][0]
#     #             select_client.delay = simdata[client_id]["endTime"]
#     #             T_client = id_to_client[client_id][1]
#     #             client_finished[client_id] = True
#     #             throughputs.append(simdata[client_id]["throughput"])

#     #             self.async_configuration([select_client], T_client)
#     #             select_client.run(reg=True)
#     #             T_cur = T_client + select_client.delay
#     #             T_new = T_cur

#     #             logging.info('Training finished on clients {} at time {} s'.format(
#     #                 select_client, T_cur
#     #             ))

#     #             # Receive client updates
#     #             reports = self.reporting([select_client])

#     #             # Update profile and plot
#     #             self.update_profile(reports)

#     #             # Perform weight aggregation
#     #             logging.info('Aggregating updates from clients {}'.format(select_client))
#     #             staleness = select_client.delay
#     #             updated_weights = self.aggregation(reports, staleness)

#     #             # Load updated weights
#     #             fl_model.load_weights(self.model, updated_weights)

#     #             # Extract flattened weights (if applicable)
#     #             if self.config.paths.reports:
#     #                 self.save_reports(round, reports)

#     #             # Save updated global model
#     #             self.async_save_model(self.model, self.config.paths.model, T_cur)

#     #             # Test global model accuracy
#     #             if self.config.clients.do_test:  # Get average accuracy from client reports
#     #                 accuracy = self.accuracy_averaging(reports)
#     #             else:  # Test updated model on server
#     #                 testset = self.loader.get_testset()
#     #                 batch_size = self.config.fl.batch_size
#     #                 testloader = fl_model.get_testloader(testset, batch_size)
#     #                 accuracy = fl_model.test(self.model, testloader)

#     #             self.throughput = 0
#     #             if len(throughputs) > 0:
#     #                 self.throughput = sum([t for t in throughputs])/len(throughputs)
#     #             logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
#     #             self.records.async_time_graphs(T_cur, accuracy, self.throughput)

#     #         # Return when target accuracy is met
#     #             if target_accuracy and \
#     #                     (self.records.get_latest_acc() >= target_accuracy):
#     #                 logging.info('Target accuracy reached.')
#     #                 break
#     #         elif simdata == 'end':
#     #             break


#     #     logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(
#     #         T_new, self.throughput
#     #     ))
#     #     cnt = 0
#     #     for c in client_finished:
#     #         if not client_finished[c]:
#     #             cnt = cnt + 1
#     #             f.write(str(c))
#     #             f.write('\n')
#     #             f.flush()

#     #     self.records.async_round_graphs(round, cnt)
#     #     return self.records.get_latest_acc(), self.records.get_latest_t()


#     def selection(self):
#         # Select devices to participate in round
#         clients_per_round = self.config.clients.per_round
#         select_type = self.config.clients.selection

#         if select_type == 'random':
#             # Select clients randomly
#             sample_clients = [client for client in random.sample(
#                 self.clients, clients_per_round)]

#         elif select_type == 'short_latency_first':
#             # Select the clients with short latencies and random loss
#             sample_clients = sorted(self.clients, key=lambda c:c.est_delay)
#             sample_clients = sample_clients[:clients_per_round]
#             print(sample_clients)

#         elif select_type == 'short_latency_high_loss_first':
#             # Get the non-negative losses and delays
#             losses = [c.loss for c in self.clients]
#             losses_norm = [l/max(losses) for l in losses]
#             delays = [c.est_delay for c in self.clients]
#             delays_norm = [d/max(losses) for d in delays]

#             # Sort the clients by jointly consider latency and loss
#             gamma = 0.2
#             sorted_idx = sorted(range(len(self.clients)),
#                                 key=lambda i: losses_norm[i] - gamma * delays_norm[i],
#                                 reverse=True)
#             print([losses[i] for i in sorted_idx])
#             print([delays[i] for i in sorted_idx])
#             sample_clients = [self.clients[i] for i in sorted_idx]
#             sample_clients = sample_clients[:clients_per_round]
#             print(sample_clients)

#         # Create one group for each selected client to perform async updates
#         #sample_groups = [Group([client]) for client in sample_clients]

#         return sample_clients

#     def async_configuration(self, sample_clients, download_time):
#         loader_type = self.config.loader
#         loading = self.config.data.loading

#         if loading == 'dynamic':
#             # Create shards if applicable
#             if loader_type == 'shard':
#                 self.loader.create_shards()

#         # Configure selected clients for federated learning task
#         for client in sample_clients:
#             if loading == 'dynamic':
#                 self.set_client_data(client)  # Send data partition to client

#             cfg_dp = getattr(self.config, "dp", None) or getattr(getattr(self.config, "fl", object()), "dp", None)
#             client.dp = cfg_dp

#             if cfg_dp is None:
#                 cfg_dp = getattr(getattr(self.config, "fl", object()), "dp", None)
#             client.dp = cfg_dp

#             logging.info(f"[DP] cfg for client {client.client_id}: {client.dp}")

#             # Extract config for client
#             config = self.config

#             # Continue configuration on client
            
#             client.async_configure(config, download_time)

#     def aggregation(self, reports, staleness=None):
#         return self.federated_async(reports, staleness)

#     def extract_client_weights(self, reports):
#         # Extract weights from reports
#         weights = [report.weights for report in reports]

#         return weights

#     def federated_async(self, reports, staleness):
#         import fl_model  # pylint: disable=import-error

#         # Extract updates from reports
#         weights = self.extract_client_weights(reports)

#         # Extract total number of samples
#         total_samples = sum([report.num_samples for report in reports])

#         # Perform weighted averaging
#         new_weights = [torch.zeros(x.size())  # pylint: disable=no-member
#                       for _, x in weights[0]]
#         for i, update in enumerate(weights):
#             num_samples = reports[i].num_samples
#             for j, (_, weight) in enumerate(update):
#                 # Use weighted average by number of samples
#                 new_weights[j] += weight * (num_samples / total_samples)

#         # Extract baseline model weights - latest model
#         baseline_weights = fl_model.extract_weights(self.model)

#         # Calculate the staleness-aware weights
#         alpha_t = self.alpha * self.staleness(staleness)
#         logging.info('{} staleness: {} alpha_t: {}'.format(
#             self.staleness_func, staleness, alpha_t
#         ))

#         # Load updated weights into model
#         updated_weights = []
#         for i, (name, weight) in enumerate(baseline_weights):
#             updated_weights.append(
#                 (name, (1 - alpha_t) * weight + alpha_t * new_weights[i])
#             )

#         return updated_weights

#     def staleness(self, staleness):
#         if self.staleness_func == "constant":
#             return 1
#         elif self.staleness_func == "polynomial":
#             a = 0.5
#             return pow(staleness+1, -a)
#         elif self.staleness_func == "hinge":
#             a, b = 10, 4
#             if staleness <= b:
#                 return 1
#             else:
#                 return 1 / (a * (staleness - b) + 1)

#     def async_save_model(self, model, path, download_time):
#         path += '/global_' + '{}'.format(download_time)
#         torch.save(model.state_dict(), path)
#         logging.info('Saved global model: {}'.format(path))

#     def rm_old_models(self, path, cur_time):
#         for filename in os.listdir(path):
#             try:
#                 model_time = float(filename.split('_')[1])
#                 if model_time < cur_time:
#                     os.remove(os.path.join(path, filename))
#                     logging.info('Remove model {}'.format(filename))
#             except Exception as e:
#                 logging.debug(e)
#                 continue

#     def update_profile(self, reports):
#         for report in reports:
#             self.profile.update(report.client_id, report.loss, report.delay,
#                                 self.flatten_weights(report.weights))