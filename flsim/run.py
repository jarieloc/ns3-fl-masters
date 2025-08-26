import argparse
import config
import logging
import os
import server
from datetime import datetime
import time


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)
    # ---- expose raw config + dp at top level (don't touch fl.dp) ----
    import json
    try:
        with open(args.config, "r") as fh:
            raw = json.load(fh)
        setattr(fl_config, "raw", raw)
        dp_block = raw.get("dp") or raw.get("federated_learning", {}).get("dp")
        if dp_block is not None:
            setattr(fl_config, "dp", dp_block)
    except Exception as e:
        logging.debug(f"DP config not attached: {e}")
# -----------------------------------------------------------------


    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        # "dqn": server.DQNServer(fl_config), # DQN server disabled
        # "dqntrain": server.DQNTrainServer(fl_config), # DQN server disabled
        "sync": server.SyncServer(fl_config),
        "async": server.AsyncServer(fl_config),
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Save and plot accuracy-time curve
    if fl_config.server in ("sync", "async"):
        import os, re

        # timestamp + basic factors
        d_str = datetime.now().strftime("%m-%d-%H-%M-%S")
        server_mode = fl_config.server
        try:
            iid = bool(fl_config.data.IID)
        except Exception:
            iid = True
        try:
            cpr = int(fl_config.clients.per_round)
        except Exception:
            cpr = 0

        # DP flag (robust to where you store it)
        try:
            dp_block = getattr(fl_config, "dp", None) or fl_config.federated_learning.dp
            dp_on = bool(dp_block["enable"] if isinstance(dp_block, dict) else dp_block.enable)
        except Exception:
            dp_on = False

        # run index: prefer array index, else parse from config filename (run01_...)
        run_idx = os.environ.get("LSB_JOBINDEX")
        if not run_idx:
            m = re.search(r"run(\d+)", os.path.basename(args.config))
            run_idx = m.group(1) if m else "00"
        run_idx = f"{int(run_idx):02d}" if str(run_idx).isdigit() else str(run_idx)

        # dataset tag -> subfolder name
        dataset = getattr(getattr(fl_config, "model", object()), "name", "Model")
        plots_root = getattr(getattr(fl_config, "paths", object()), "plot", "./plots")
        subdir = "plotsMNIST" if dataset.upper().startswith("MNIST") or dataset == "MNIST" else f"plots{dataset}"
        out_dir = os.path.join(plots_root, subdir)
        os.makedirs(out_dir, exist_ok=True)

        # filename with factors
        base = f"run{run_idx}_{server_mode}_{'IID' if iid else 'nonIID'}_c{cpr}_{'DPon' if dp_on else 'DPoff'}_{d_str}"

        fl_server.records.save_record(os.path.join(out_dir, base + ".csv"))
        fl_server.records.plot_record(os.path.join(out_dir, base + ".png"))

    # # Save and plot accuracy-time curve
    # if fl_config.server == "sync" or fl_config.server == "async":
    #     d_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    #     network_type = fl_config.network.type
    #     total_clients = str(fl_config.clients.total)
    #     per_round = str(fl_config.clients.per_round)

    #     fl_server.records.save_record('{}_{}_{}_{}outOf{}.csv'.format(
    #         fl_config.server, d_str, network_type, per_round, total_clients
    #     ))
    #     fl_server.records.plot_record('{}_{}_{}_{}outOf{}.png'.format(
    #         fl_config.server, d_str, network_type, per_round, total_clients
    #     ))

    # Delete global model
    #os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    st = time.time()
    main()
    elapsed = time.time() - st
    logging.info('The program takes {} s'.format(
        time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    ))
