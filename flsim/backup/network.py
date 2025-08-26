
# flsim/network.py — ns-3 (terasim) aware adapter
# - Wi-Fi/Ethernet: socket-only via wifi_exp binary, binary protocol
# - THz (6G): runs scratch app and parses a final JSON summary line
from __future__ import annotations
import json, os, shlex, socket, struct, subprocess, time, pathlib
from typing import Any, Dict, Iterable, List, Optional, Tuple
import tempfile

# --- Wire protocol (must match fl-sim-interface.*) ---
CMD_RESPONSE       = 0
CMD_RUN_SIMULATION = 1
CMD_EXIT           = 2
# Request header: <II>  (command, nItems)
# For Wi-Fi/Ethernet request: nItems * <I>  (bitmap flags)
# Response: <II> + nItems * <Qdd>  (id:uint64, roundTime:double, throughput:double)

# ------------------------ Utilities ------------------------

def _log(msg: str) -> None:
    print(f"[NET] {msg}", flush=True)

def _as_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    raw = getattr(cfg, "raw", None)
    if isinstance(raw, dict):
        return raw
    d: Dict[str, Any] = {}
    net = getattr(cfg, "network", None)
    if net:
        d["network"] = {
            "type": getattr(net, "type", "wifi"),
            "controller": getattr(net, "controller", "socket"),
            "socket": getattr(net, "socket", {"host":"127.0.0.1","port":8080,"launch":True}),
            "wifi": getattr(net, "wifi", {}) or {},
            "ethernet": getattr(net, "ethernet", {}) or {},
            "thz": getattr(net, "thz", {}) or {},
            "ns3_root": getattr(net, "ns3_root", None),
        }
    clients = getattr(cfg, "clients", None)
    if clients:
        d["clients"] = {"total": int(getattr(clients, "total", 1)),
                        "per_round": int(getattr(clients, "per_round", 1))}
    model = getattr(cfg, "model", None)
    if model:
        d["model"] = {"name": getattr(model, "name", "MNIST"),
                      "size": int(getattr(model, "size", 0))}  # KB
    d["server"] = getattr(cfg, "server", "async")
    return d

def _resolve_ns3_root(cfg: Dict[str, Any]) -> pathlib.Path:
    override = (cfg.get("network") or {}).get("ns3_root")
    if override:
        p = pathlib.Path(override).expanduser().resolve()
        if p.exists():
            return p

    env = os.environ.get("NS3_ROOT")
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if p.exists():
            return p

    here = pathlib.Path(__file__).resolve()
    candidates = [
        here.parents[1] / "ns3-fl-network",
        here.parents[1] / "ns3",
        here.parents[2] / "ns3-fl-network",
        here.parents[2] / "ns3",
        here.parents[3] / "ns3-fl-network",
        here.parents[3] / "ns3",
    ]
    for c in candidates:
        if (c / "ns3").exists() or (c / "contrib").exists() or (c / "build").exists() or (c / "CMakeLists.txt").exists():
            return c
    tried = [str(x) for x in candidates]
    raise FileNotFoundError(
        "Could not locate your ns-3 tree. Set network.ns3_root in config or NS3_ROOT env var.\n"
        "Tried:\n  - " + "\n  - ".join(tried)
    )

# ------------------------ Adapter ------------------------

class Network:
    def __init__(self, cfg_any: Any):
        cfg = _as_dict(cfg_any)
        self.cfg = cfg
        net = cfg.get("network", {}) or {}
        self.network_type = str(net.get("type", "wifi")).lower()
        self.controller   = str(net.get("controller", "socket")).lower()
        self.num_clients  = int((cfg.get("clients") or {}).get("total", 1))
        self.per_round    = int((cfg.get("clients") or {}).get("per_round", 1))

        model_cfg = cfg.get("model", {}) or {}
        self.model_size_kb = int(model_cfg.get("size", 0))               # KB
        self.model_bytes   = self.model_size_kb * 1024                   # bytes (THz uses bytes)

        sock = net.get("socket", {}) or {}
        self.host   = str(sock.get("host", "127.0.0.1"))
        self.port   = int(sock.get("port", 8080))
        self.launch = bool(sock.get("launch", True))

        self.wifi_cfg = net.get("wifi", {}) or {}
        self.eth_cfg  = net.get("ethernet", {}) or {}
        self.thz_cfg  = net.get("thz", {}) or {}

        self._use_thz = (self.network_type == "thz")
        self._sock: Optional[socket.socket] = None
        self._child: Optional[subprocess.Popen] = None
        self._child_log = None

        # Locate ns-3 (terasim layout) and the ./ns3 helper
        self.NS3_ROOT = _resolve_ns3_root(cfg)
        self.NS3_CLI  = str(self.NS3_ROOT / "ns3")   # terasim helper

        self._pending_sel: Optional[List[int]] = None

        mode = "THz(JSON inline)" if self._use_thz else f"Wi-Fi/Ethernet({self.controller})"
        _log(f"initialized | mode={mode} | host={self.host} port={self.port} | ns3_root={self.NS3_ROOT}")

        if not self._use_thz and self.launch and self.controller == "socket":
            self._launch_socket_child()

    # ---------------- Small helpers ----------------

    @staticmethod
    def _items_to_dict(items: List[Tuple[int, float, float]]) -> Dict[int, Dict[str, float]]:
        return {int(eid): {"roundTime": float(rt), "throughput": float(thr)}
                for (eid, rt, thr) in items}

    @staticmethod
    def _parse_ids(arr: Iterable[Any]) -> List[int]:
        ids: List[int] = []
        for x in arr:
            if hasattr(x, "client_id"):
                ids.append(int(getattr(x, "client_id")))
            else:
                ids.append(int(x))
        return ids

    # Maintain API used by asyncServer
    def parse_clients(self, sample_clients: Iterable[Any]) -> List[int]:
        return self._parse_ids(sample_clients)

    # ---------------- Lifecycle ----------------

    def _socket_send_round(self, requestType: int, ids: list) -> list:
        """
        Do one sync socket round and return a list of dicts with
        id/startTime/endTime/throughput for each returned client.
        """
        if self._sock is None:
            self.connect()

        n = int(self.num_clients)
        sel = set(int(x) for x in ids)

        # Request header + bitmap
        hdr = struct.pack("<II", CMD_RUN_SIMULATION, n)
        flags = [1 if i in sel else 0 for i in range(n)]
        body = struct.pack("<" + "I"*n, *flags)
        self._send_all(hdr + body)

        # Response is n × (id, roundTime, throughput)
        items = self._recv_response(timeout=180.0)  # [(eid, rt, thr), ...]
        out = []
        for (eid, rt, thr) in items:
            if int(eid) in sel:
                out.append({
                    "id": int(eid),
                    "startTime": 0.0,
                    "endTime": float(rt),
                    "throughput": float(thr)
                })
        return out

    def connect(self) -> bool:
        # No socket needed for THz or for inline controller
        if self._use_thz or self.controller != "socket":
            return True

        if self._child is not None and self._child.poll() is not None:
            _log(f"child exited rc={self._child.returncode}; relaunching")
            self._launch_socket_child()
        try:
            if self._sock is None:
                self._connect_socket_with_retries((self.host, self.port), retries=80, delay=0.125)
        except Exception as e:
            tail = ""
            try:
                if getattr(self, "_child_log", None):
                    self._child_log.flush()
                    with open(self._child_log.name, "r") as f:
                        tail = "".join(f.readlines()[-100:])
            except Exception:
                pass
            rc = self._child.poll() if self._child else None
            raise RuntimeError(f"[ns3] connect failed (rc={rc}): {e}\n--- wifi_exp.log tail ---\n{tail}\n---")
        _log(f"connected | child={'up' if self._child and self._child.poll() is None else 'down'} sock={bool(self._sock)}")
        return True


    # def connect(self) -> bool:
    #     if self._use_thz:
    #         return True
    #     if self._child is not None and self._child.poll() is not None:
    #         _log(f"child exited rc={self._child.returncode}; relaunching")
    #         self._launch_socket_child()
    #     try:
    #         if self._sock is None:
    #             self._connect_socket_with_retries((self.host, self.port), retries=80, delay=0.125)
    #     except Exception as e:
    #         tail = ""
    #         try:
    #             if getattr(self, "_child_log", None):
    #                 self._child_log.flush()
    #                 with open(self._child_log.name, "r") as f:
    #                     tail = "".join(f.readlines()[-100:])
    #         except Exception:
    #             pass
    #         rc = self._child.poll() if self._child else None
    #         raise RuntimeError(f"[ns3] connect failed (rc={rc}): {e}\n--- wifi_exp.log tail ---\n{tail}\n---")
    #     _log(f"connected | child={'up' if self._child and self._child.poll() is None else 'down'} sock={bool(self._sock)}")
    #     return True

    def disconnect(self) -> None:
        # Politely tell C++ to stop, then close socket
        try:
            if self._sock:
                try: self._sock.sendall(struct.pack("<II", CMD_EXIT, 0))
                except Exception: pass
                try: self._sock.shutdown(socket.SHUT_RDWR)
                except Exception: pass
                self._sock.close()
        finally:
            self._sock = None

        # Reap the child if still running
        if self._child is not None:
            try:
                if self._child.poll() is None:
                    _log("terminating child...")
                    self._child.terminate()
                    try:
                        self._child.wait(timeout=3.0)
                    except Exception:
                        _log("killing child...")
                        try: self._child.kill()
                        except Exception:
                            pass
            finally:
                self._child = None
        _log("disconnected")

    # ---------------- Server API ----------------

    def sendRequest(self, requestType: int, array: Iterable[int]) -> Dict[int, Dict[str, float]]:
        active_ids = self._parse_ids(array)
        if not active_ids:
            return {}

        if self._use_thz:
            items = self._run_thz(active_ids)
            data = self._items_to_dict(items)
            sel = set(active_ids)
            return {i: data[i] for i in data if i in sel}
        
        if not self._use_thz and self.controller != "socket":
            raise NotImplementedError("Wi-Fi/Ethernet in 'inline' mode is not implemented in this network.py. Use controller: 'socket'.")


        if self._sock is None:
            self.connect()
        if requestType != CMD_RUN_SIMULATION:
            raise ValueError(f"unsupported requestType={requestType}")

        # Header: nItems must equal total number of clients on the C++ side
        n = int(self.num_clients)
        hdr = struct.pack("<II", CMD_RUN_SIMULATION, n)

        # Body: bitmap n * <I> flags (1 if selected, else 0)
        sel = set(int(x) for x in active_ids)
        flags = [1 if i in sel else 0 for i in range(n)]
        body = struct.pack("<" + "I"*n, *flags)
        self._send_all(hdr + body)

        items = self._recv_response(timeout=180.0)
        data = self._items_to_dict(items)
        return {i: data[i] for i in data if i in sel}
    

    def sendAsyncRequest(self, requestType: int, array):
        """
        Queue-based async:
          - THz: spawn ns-3 and return immediately; results parsed later
          - Socket (Wi-Fi/Ethernet): unchanged (send flags now; read later)
        """
        
        active_ids = self._parse_ids(array)
        self._async_ids = list(active_ids)
        self._async_queue = []
        self._deadline = None

        if not active_ids:
            self._proc = None
            return

        # --- THz: spawn a subprocess and return immediately ---
        if self._use_thz:
            cmd = self._thz_cmd(active_count=len(active_ids))
            _log("[thz] spawn: " + " ".join(shlex.quote(x) for x in cmd))

            # quiet ns-3 log spam so the run finishes quickly and we can parse JSON
            env = os.environ.copy()
            env["NS_LOG"] = ""  # disable NS_LOG chatter

            # log to a file (no pipe back-pressure)
            self._thz_log_path = pathlib.Path(self.NS3_ROOT) / "thz_run.log"
            self._thz_log_file = open(self._thz_log_path, "w", buffering=1)
            self._proc = subprocess.Popen(
                cmd, cwd=str(self.NS3_ROOT), env=env,
                stdout=self._thz_log_file, stderr=self._thz_log_file, text=True
            )

            # optional timeout (seconds) from config.network.thz.timeout_sec; default 120
            self._deadline = time.time() + float(self.thz_cfg.get("timeout_sec", 120))
            return

        if not self._use_thz and self.controller != "socket":
            raise NotImplementedError("Wi-Fi/Ethernet in 'inline' mode is not implemented in this network.py. Use controller: 'socket'.")

        # --- Socket path (Wi-Fi/Ethernet): your existing sender ---
        if self._sock is None:
            self.connect()
        if requestType != CMD_RUN_SIMULATION:
            raise ValueError(f"unsupported requestType={requestType}")

        n = int(self.num_clients)
        hdr = struct.pack("<II", CMD_RUN_SIMULATION, n)
        sel = set(int(x) for x in active_ids)
        flags = [1 if i in sel else 0 for i in range(n)]
        body = struct.pack("<" + "I"*n, *flags)
        self._send_all(hdr + body)

        # remember selection so we can filter when the response arrives
        self._pending_sel = list(active_ids)


    # def sendAsyncRequest(self, requestType: int, array):
    #     """
    #     Queue-based async:
    #     - THz: run once and queue results after the process ends (same as before)
    #     - Socket (Wi-Fi/Ethernet): do one sync round now and prefill a queue
    #     """
    #     active_ids = self._parse_ids(array)
    #     self._async_ids = list(active_ids)
    #     self._async_queue = []

    #     if not active_ids:
    #         return

    #     # THz path: keep your existing spawn+deadline behavior if you use it
    #     if self._use_thz:
    #         res = self._run_thz(active_ids)  # [(id, rt, thr), ...]
    #         for (cid, rt, thr) in res:
    #             self._async_queue.append({
    #                 int(cid): {"startTime": 0.0, "endTime": float(rt), "throughput": float(thr)}
    #             })
    #         return

    #     # Socket controller path: one blocking round now → fill queue
    #     res = self._socket_send_round(requestType=requestType, ids=active_ids)
    #     for e in res:
    #         cid = int(e["id"])
    #         self._async_queue.append({
    #             cid: {
    #                 "startTime": float(e["startTime"]),
    #                 "endTime": float(e["endTime"]),
    #                 "throughput": float(e["throughput"]),
    #             }
    #         })

    # def sendAsyncRequest(self, requestType: int, array: Iterable[int]) -> None:
    #     active_ids = self._parse_ids(array)
    #     if not active_ids:
    #         self._pending_sel = []
    #         return

    #     if self._use_thz:
    #         # Defer THz run until readAsyncResponse
    #         self._pending_sel = active_ids
    #         return

    #     if self._sock is None:
    #         self.connect()
    #     if requestType != CMD_RUN_SIMULATION:
    #         raise ValueError(f"unsupported requestType={requestType}")

    #     n = int(self.num_clients)
    #     hdr = struct.pack("<II", CMD_RUN_SIMULATION, n)

    #     sel = set(int(x) for x in active_ids)
    #     flags = [1 if i in sel else 0 for i in range(n)]
    #     body = struct.pack("<" + "I"*n, *flags)

    #     self._send_all(hdr + body)
    #     self._pending_sel = list(active_ids)

    def readAsyncResponse(self, timeout: float = 0.0):
        """
        Non-blocking:
          - THz: returns {} while ns-3 runs; after finish, emits one {id:{...}} per call; then 'end'
          - Socket: read from socket once and translate to {id:{...}} dicts (unchanged semantics)
        """
        # --- THz path ---
        if self._use_thz:
            # nothing started and nothing queued
            if self._proc is None and not self._async_queue:
                return 'end'

            # process still running?
            if self._proc is not None and self._proc.poll() is None:
                # enforce optional deadline
                if self._deadline and time.time() > self._deadline:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                    try:
                        self._proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            self._proc.kill()
                        except Exception:
                            pass
                return {}

            # finished: build queue from the last JSON line in the log
            # finished: build queue from the last JSON line in the log
            if self._proc is not None:
                try:
                    if getattr(self, "_thz_log_file", None):
                        self._thz_log_file.flush()
                        self._thz_log_file.close()
                finally:
                    self._thz_log_file = None

                self._proc = None  # free handle

                # parse last JSON line
                last = None
                if getattr(self, "_thz_log_path", None) and self._thz_log_path.exists():
                    with open(self._thz_log_path, "r") as fh:
                        for line in reversed(fh.read().splitlines()):
                            s = line.strip()
                            if s.startswith("{") and s.endswith("}"):
                                last = s
                                break

                self._async_queue = []
                if last:
                    data = json.loads(last)
                    id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
                    for e in data.get("clientResults", []):
                        local = int(e.get("id", -1))
                        if local not in id_map:
                            continue
                        real = id_map[local]
                        done = float(e.get("doneAt", 0.0))
                        rx   = float(e.get("rxBytes", 0.0))
                        thr  = (rx / done) if done > 0 else 0.0
                        self._async_queue.append({
                            int(real): {
                                "startTime": 0.0,
                                "endTime":   done,
                                "roundTime": done,     # ← add roundTime for server fallback
                                "throughput": thr
                            }
                        })
                else:
                    # Fallback if the sim was killed before printing JSON
                    sim_t = float(self.thz_cfg.get("sim_time", 0.5))
                    for real in self._async_ids:
                        self._async_queue.append({
                            int(real): {
                                "startTime": 0.0,
                                "endTime":   sim_t,
                                "roundTime": sim_t,    # ← so _handle_one_result can use it
                                "throughput": 0.0
                            }
                        })

                # consume ids now that we've queued results (or fallback estimates)
                self._async_ids = []


            # if self._proc is not None:
            #     try:
            #         if self._thz_log_file:
            #             self._thz_log_file.flush()
            #             self._thz_log_file.close()
            #     finally:
            #         self._thz_log_file = None

            #     self._proc = None  # free handle

            #     # parse last JSON line
            #     last = None
            #     if self._thz_log_path and self._thz_log_path.exists():
            #         with open(self._thz_log_path, "r") as fh:
            #             for line in reversed(fh.read().splitlines()):
            #                 s = line.strip()
            #                 if s.startswith("{") and s.endswith("}"):
            #                     last = s
            #                     break
            #     if last:
            #         data = json.loads(last)
            #         id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
            #         for e in data.get("clientResults", []):
            #             local = int(e.get("id", -1))
            #             if local not in id_map:
            #                 continue
            #             real = id_map[local]
            #             done = float(e.get("doneAt", 0.0))
            #             rx   = float(e.get("rxBytes", 0.0))
            #             thr  = (rx / done) if done > 0 else 0.0
            #             self._async_queue.append({
            #                 int(real): {"startTime": 0.0, "endTime": done, "throughput": thr}
            #             })

            # deliver one dict per call
            if self._async_queue:
                return self._async_queue.pop(0)
            return 'end'

        # --- Socket path (unchanged): return dicts keyed by id ---
        if getattr(self, "_async_queue", None):
            return self._async_queue.pop(0)

        # If we have a pending selection and haven't read yet, read exactly once
        if getattr(self, "_pending_sel", None):
            sel_set = set(int(x) for x in self._pending_sel)
            items = self._recv_response(timeout=180.0)  # [(eid, rt, thr), ...]
            data = self._items_to_dict(items)           # {id: {'roundTime':..., 'throughput':...}}
            # build a per-client queue filtered to the selection
            self._async_queue = [{cid: data[cid]} for cid in data if (not sel_set) or cid in sel_set]
            self._pending_sel = None
            if self._async_queue:
                return self._async_queue.pop(0)
            return 'end'

        # Nothing in flight
        return 'end'



    # def readAsyncResponse(self, timeout: float = 0.0):
    #     """
    #     Non-blocking:
    #     - If there are queued items, deliver one {id: {...}} dict
    #     - When queue is empty, return 'end'
    #     """
    #     q = getattr(self, "_async_queue", None)
    #     if not q:
    #         return 'end'
    #     return q.pop(0) if q else 'end'

    # def readAsyncResponse(self, timeout: float = 180.0) -> Dict[int, Dict[str, float]]:
    #     sel = set(int(x) for x in (self._pending_sel or []))

    #     if self._use_thz:
    #         if not sel:
    #             return {}
    #         items = self._run_thz(list(sel))
    #         self._pending_sel = None
    #         return self._items_to_dict(items)

    #     items = self._recv_response(timeout=timeout)  # [(eid, rt, thr), ...]
    #     self._pending_sel = None
    #     data = self._items_to_dict(items)
    #     return {i: data[i] for i in data if (not sel) or i in sel}

    def readSyncResponse(self, timeout: float = 180.0) -> Dict[int, Dict[str, float]]:
        return self._items_to_dict(self._recv_response(timeout=timeout))

    # ---------------- THz (JSON via ./ns3 run) ----------------

    def _thz_cmd(self, active_count: int) -> List[str]:
        args = [
            self.NS3_CLI, "run", "scratch/thz-macro-central", "--",
            f"--nodeNum={self.num_clients}",
            f"--clients={active_count}",
            f"--modelBytes={self.model_bytes}",
        ]
        for k, fmt in [
            ("carrier_ghz", "--CarrierFreqGHz={}"),
            ("pkt_size",    "--pktSize={}"),
            ("sim_time",    "--simTime={}"),
            ("interval_us", "--intervalUs={}"),
            ("way",         "--way={}"),
            ("radius",      "--radius={}"),
            ("beamwidth",   "--beamwidth={}"),
            ("gain",        "--gain={}"),
            ("ap_angle",    "--apAngle={}"),
            ("sta_angle",   "--staAngle={}"),
            ("useWhiteList","--useWhiteList={}"),
        ]:
            if k in self.thz_cfg:
                args.append(fmt.format(self.thz_cfg[k]))
        return args


    # --- inside flsim/network.py, replace def _run_thz(...) with: ---
    def _run_thz(self, active_ids):
        """
        Run ns-3 THz scratch and stream stdout so we don't block forever.
        Returns: [(real_id, roundTime, throughput), ...]
        """
        import shlex, json, time, subprocess, os

        # argv
        args = [
            self.NS3_CLI, "run", "scratch/thz-macro-central", "--",
            f"--nodeNum={self.num_clients}",
            f"--clients={len(active_ids)}",
            f"--modelBytes={self.model_bytes}",
        ]
        thz_cfg = getattr(self, "thz_cfg", getattr(self, "_thz_cfg", {})) or {}
        for k, fmt in [
            ("carrier_ghz", "--CarrierFreqGHz={}"),
            ("pkt_size",    "--pktSize={}"),
            ("sim_time",    "--simTime={}"),
            ("interval_us", "--intervalUs={}"),
            ("way",         "--way={}"),
            ("radius",      "--radius={}"),
            ("beamwidth",   "--beamwidth={}"),
            ("gain",        "--gain={}"),
            ("ap_angle",    "--apAngle={}"),
            ("sta_angle",   "--staAngle={}"),
            ("useWhiteList","--useWhiteList={}"),
        ]:
            if k in thz_cfg:
                args.append(fmt.format(thz_cfg[k]))

        _log("[thz] " + " ".join(shlex.quote(x) for x in args))

        # Stream stdout; mute ns-3 logs
        env = os.environ.copy()
        env["NS_LOG"] = ""
        p = subprocess.Popen(
            args, cwd=str(self.NS3_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env
        )

        last_json_line = None

        # SHORT deadline: honor config.network.thz.timeout_sec if present
        sim_time = float(thz_cfg.get("sim_time", 0.8))
        timeout_sec = float(thz_cfg.get("timeout_sec", max(5.0, 5.0 * sim_time)))
        deadline = time.time() + timeout_sec

        assert p.stdout is not None
        for line in p.stdout:
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                last_json_line = s
            if time.time() > deadline:
                try: p.kill()
                except Exception: pass
                break

        try:
            p.wait(timeout=3.0)
        except Exception:
            try: p.kill()
            except Exception: pass

        # If no JSON came back, synthesize a sensible fallback
        if last_json_line is None:
            rt = sim_time if sim_time > 0 else 1.0
            thr = self.model_bytes / max(1e-6, rt)
            return [(int(active_ids[i]), float(rt), float(thr)) for i in range(len(active_ids))]

        data = json.loads(last_json_line)
        id_map = {local: active_ids[local] for local in range(len(active_ids))}
        items = []
        for e in data.get("clientResults", []):
            local = int(e.get("id", -1))
            if local not in id_map:
                continue
            real = int(id_map[local])
            rt   = float(e.get("doneAt", e.get("roundTime", 0.0)) or 0.0)
            rx   = float(e.get("rxBytes", 0.0))
            thr  = (rx / rt) if rt > 0 else 0.0
            items.append((real, rt, thr))
        if not items:
            rt = sim_time if sim_time > 0 else 1.0
            thr = self.model_bytes / max(1e-6, rt)
            items = [(int(active_ids[i]), float(rt), float(thr)) for i in range(len(active_ids))]
        return items


    # def _run_thz(self, active_ids):
    #     """
    #     Run ns-3 THz scratch and stream stdout so we don't block forever.
    #     Returns: [(real_id, roundTime, throughput), ...]
    #     """
    #     import shlex, json, time, subprocess

    #     # Build argv exactly as you already do
    #     args = [
    #         self.NS3_CLI, "run", "scratch/thz-macro-central", "--",
    #         f"--nodeNum={self.num_clients}",
    #         f"--clients={len(active_ids)}",
    #         f"--modelBytes={self.model_bytes}",
    #     ]
    #     thz_cfg = getattr(self, "thz_cfg", getattr(self, "_thz_cfg", {})) or {}
    #     for k, fmt in [
    #         ("carrier_ghz", "--CarrierFreqGHz={}"),
    #         ("pkt_size",    "--pktSize={}"),
    #         ("sim_time",    "--simTime={}"),
    #         ("interval_us", "--intervalUs={}"),
    #         ("way",         "--way={}"),
    #         ("radius",      "--radius={}"),
    #         ("beamwidth",   "--beamwidth={}"),
    #         ("gain",        "--gain={}"),
    #         ("ap_angle",    "--apAngle={}"),
    #         ("sta_angle",   "--staAngle={}"),
    #         ("useWhiteList","--useWhiteList={}"),
    #     ]:
    #         if k in thz_cfg:
    #             args.append(fmt.format(thz_cfg[k]))

    #     _log("[thz] " + " ".join(shlex.quote(x) for x in args))

    #     # Stream stdout; drop stderr noise
    #     p = subprocess.Popen(
    #         args, cwd=str(self.NS3_ROOT),
    #         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    #         text=True, bufsize=1
    #     )

    #     last_json_line = None
    #     # Short wall-clock guard: 30s or 10×sim_time, whichever is larger
    #     sim_time = float(thz_cfg.get("sim_time", 0.8))
    #     deadline = time.time() + max(30.0, 10.0 * sim_time)

    #     assert p.stdout is not None
    #     for line in p.stdout:
    #         s = line.strip()
    #         if s.startswith("{") and s.endswith("}"):
    #             last_json_line = s
    #         if time.time() > deadline:
    #             try:
    #                 p.kill()
    #             finally:
    #                 break

    #     rc = p.wait()

    #     # If no JSON came back, synthesize something sensible so FL can proceed
    #     if last_json_line is None:
    #         rt = sim_time if sim_time > 0 else 1.0
    #         # conservative: assume model transmitted in 'rt' seconds
    #         thr = self.model_bytes / rt
    #         return [(int(active_ids[i]), float(rt), float(thr)) for i in range(len(active_ids))]

    #     data = json.loads(last_json_line)

    #     id_map = {local: active_ids[local] for local in range(len(active_ids))}
    #     items = []
    #     for e in data.get("clientResults", []):
    #         local = int(e.get("id", -1))
    #         if local not in id_map:
    #             continue
    #         real = int(id_map[local])
    #         rt = float(e.get("doneAt", e.get("roundTime", 0.0)) or 0.0)
    #         rx = float(e.get("rxBytes", 0.0))
    #         thr = (rx / rt) if rt > 0 else 0.0
    #         items.append((real, rt, thr))
    #     return items


    # ---------------- Wi-Fi/Ethernet child (wifi_exp) ----------------

    @staticmethod
    def _find_wifi_bin_path(ns3_root: pathlib.Path) -> Optional[pathlib.Path]:
        for d in (
            ns3_root / "build" / "bin",
            ns3_root / "cmake-cache" / "build" / "bin",
            ns3_root / "cmake-cache" / "bin",
        ):
            for p in d.glob("ns*-wifi_exp*"):
                if p.is_file() and os.access(p, os.X_OK):
                    return p
        return None

    def _launch_socket_child(self) -> None:
        bin_override = (self.wifi_cfg.get("binary") if self.network_type == "wifi"
                        else self.eth_cfg.get("binary"))
        wifi_bin = pathlib.Path(bin_override).resolve() if bin_override else self._find_wifi_bin_path(self.NS3_ROOT)
        if not wifi_bin or not wifi_bin.exists():
            _log("[ns3] building (wifi_exp missing)...")
            self._ns3_build()
            wifi_bin = pathlib.Path(bin_override).resolve() if bin_override else self._find_wifi_bin_path(self.NS3_ROOT)
        if not wifi_bin or not wifi_bin.exists():
            raise FileNotFoundError("wifi_exp executable not found after build.")

        argv = [str(wifi_bin)] + self._wifi_args()
        _log("launch(bin): " + " ".join(shlex.quote(x) for x in argv))

        env = os.environ.copy()
        sep = ":" if os.name != "nt" else ";"
        libdir = self.NS3_ROOT / "build" / "lib"
        bindir = self.NS3_ROOT / "build" / "bin"
        pydir  = self.NS3_ROOT / "build" / "bindings" / "python"

        if bindir.exists():
            env["PATH"] = str(bindir) + (sep + env.get("PATH", ""))
        if os.name != "nt" and libdir.exists():
            env["LD_LIBRARY_PATH"] = str(libdir) + (sep + env.get("LD_LIBRARY_PATH", ""))
        if pydir.exists():
            env["PYTHONPATH"] = str(pydir) + (sep + env.get("PYTHONPATH", ""))

        log_path = pathlib.Path(self.NS3_ROOT) / "wifi_exp.launch.log"
        self._child_log = open(log_path, "a+", buffering=1)
        self._child = subprocess.Popen(
            argv, cwd=str(self.NS3_ROOT), env=env,
            stdout=self._child_log, stderr=self._child_log, text=True
        )
        _log(f"child started (pid={self._child.pid}); logging → {log_path}")
        time.sleep(0.3)

    def _ns3_build(self) -> None:
        if not self.NS3_ROOT.exists():
            raise FileNotFoundError(f"ns-3 tree not found at {self.NS3_ROOT}")
        proc = subprocess.run(self.NS3_CLI + " build", cwd=str(self.NS3_ROOT),
                              shell=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ns-3 build failed:\n{proc.stderr}")

    def _wifi_args(self) -> List[str]:
        nic = self.wifi_cfg if self.network_type == "wifi" else self.eth_cfg
        args = [
            f"--NumClients={self.num_clients}",
            f"--NetworkType={'wifi' if self.network_type == 'wifi' else 'ethernet'}",
            f"--MaxPacketSize={int(nic.get('max_packet_size', 1024))}",
            f"--TxGain={float(nic.get('tx_gain', 0.0))}",
            f"--ModelSize={int(self.model_size_kb)}",   # KB
            f"--LearningModel={'sync'}",
            f"--UseFLSocket={'true' if self.controller=='socket' else 'false'}",
        ]
        if "data_rate" in nic:
            args.append(f"--DataRate={str(nic['data_rate'])}")
        if "sim_time" in nic:
            args.append(f"--SimTime={float(nic['sim_time'])}")
        return args
    
    # def _wifi_args(self) -> List[str]:
    #     nic = self.wifi_cfg if self.network_type == "wifi" else self.eth_cfg
    #     args = [
    #         f"--NumClients={self.num_clients}",
    #         f"--NetworkType={'wifi' if self.network_type == 'wifi' else 'ethernet'}",
    #         f"--MaxPacketSize={int(nic.get('max_packet_size', 1024))}",
    #         f"--TxGain={float(nic.get('tx_gain', 0.0))}",
    #         f"--ModelSize={int(self.model_size_kb)}",   # KB
    #         "--LearningModel=sync",
    #         "--UseFLSocket=true",
    #     ]
    #     if "data_rate" in nic:
    #         args.append(f"--DataRate={str(nic['data_rate'])}")
    #     if "sim_time" in nic:
    #         args.append(f"--SimTime={float(nic['sim_time'])}")
    #     return args

    # ---------------- Socket helpers ----------------

    def _send_all(self, payload: bytes) -> None:
        try:
            self._sock.sendall(payload)  # type: ignore[union-attr]
        except (BrokenPipeError, ConnectionResetError):
            _log("send failed; attempting reconnect")
            self._connect_socket_with_retries((self.host, self.port), retries=40, delay=0.25)
            self._sock.sendall(payload)  # type: ignore[union-attr]

    def _recv_all(self, n: int, timeout: float = 60.0) -> bytes:
        if self._sock is None:
            raise RuntimeError("[NET][socket] not connected")
        self._sock.settimeout(timeout)
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                rc = self._child.returncode if (self._child and self._child.poll() is not None) else None
                tail = ""
                try:
                    if self._child_log:
                        self._child_log.flush()
                        with open(self._child_log.name, "r") as f:
                            tail = "".join(f.readlines()[-60:])
                except Exception:
                    pass
                raise RuntimeError(f"[NET][socket] connection closed (child rc={rc}). ns-3 log tail:\n{tail}")
            buf.extend(chunk)
        return bytes(buf)

    def _recv_response(self, timeout: float) -> List[Tuple[int, float, float]]:
        hdr = self._recv_all(8, timeout=timeout)
        (cmd, n) = struct.unpack("<II", hdr)
        if cmd != CMD_RESPONSE:
            raise RuntimeError(f"[NET][socket] unexpected cmd={cmd}, expected RESPONSE=0")
        out: List[Tuple[int, float, float]] = []
        for _ in range(n):
            blob = self._recv_all(24, timeout=timeout)  # Qdd
            (eid, rt, thr) = struct.unpack("<Qdd", blob)
            out.append((int(eid), float(rt), float(thr)))
        return out

    def _connect_socket_with_retries(self, addr: Tuple[str, int], retries: int, delay: float):
        (host, port) = addr
        last_err: Optional[BaseException] = None
        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((host, port))
                self._sock = s
                _log(f"socket connected to {host}:{port} (attempt {i+1}/{retries})")
                return
            except Exception as e:
                last_err = e
                time.sleep(delay)
        raise ConnectionError(f"could not connect to {host}:{port} after {retries} attempts: {last_err}")

# # flsim/network.py — ns-3 (terasim) aware adapter
# # - Wi-Fi/Ethernet: socket-only via "./ns3 run wifi_exp -- ...", binary protocol
# # - THz (6G): "./ns3 run scratch/thz-macro-central -- ...", parse final JSON line
# from __future__ import annotations
# import json, os, shlex, socket, struct, subprocess, time, pathlib
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# # --- Wire protocol (must match your fl-sim-interface.*) ---
# CMD_RESPONSE       = 0
# CMD_RUN_SIMULATION = 1
# CMD_EXIT           = 2
# # Request header: <II>  (command, nItems)
# # For Wi-Fi/Ethernet request: nItems * <I>  (client IDs, little-endian)
# # Response: <II> + nItems * <Qdd>  (id:uint64, roundTime:double, throughput:double)

# # ------------------------ Utilities ------------------------

# def _log(msg: str) -> None:
#     print(f"[NET] {msg}", flush=True)

# def _as_dict(cfg: Any) -> Dict[str, Any]:
#     if isinstance(cfg, dict):
#         return cfg
#     raw = getattr(cfg, "raw", None)
#     if isinstance(raw, dict):
#         return raw
#     d: Dict[str, Any] = {}
#     net = getattr(cfg, "network", None)
#     if net:
#         d["network"] = {
#             "type": getattr(net, "type", "wifi"),
#             "controller": getattr(net, "controller", "socket"),
#             "socket": getattr(net, "socket", {"host":"127.0.0.1","port":8080,"launch":True}),
#             "wifi": getattr(net, "wifi", {}) or {},
#             "ethernet": getattr(net, "ethernet", {}) or {},
#             "thz": getattr(net, "thz", {}) or {},
#             "ns3_root": getattr(net, "ns3_root", None),
#         }
#     clients = getattr(cfg, "clients", None)
#     if clients:
#         d["clients"] = {"total": int(getattr(clients, "total", 1)),
#                         "per_round": int(getattr(clients, "per_round", 1))}
#     model = getattr(cfg, "model", None)
#     if model:
#         d["model"] = {"name": getattr(model, "name", "MNIST"),
#                       "size": int(getattr(model, "size", 0))}  # KB
#     d["server"] = getattr(cfg, "server", "async")
#     return d

# def _resolve_ns3_root(cfg: Dict[str, Any]) -> pathlib.Path:
#     # 1) explicit override in config
#     override = (cfg.get("network") or {}).get("ns3_root")
#     if override:
#         p = pathlib.Path(override).expanduser().resolve()
#         if p.exists():
#             return p

#     # 2) environment variable
#     env = os.environ.get("NS3_ROOT")
#     if env:
#         p = pathlib.Path(env).expanduser().resolve()
#         if p.exists():
#             return p

#     # 3) auto-discover relative to this file
#     here = pathlib.Path(__file__).resolve()
#     # parents[1] == .../ns-fl      (your case)
#     # parents[2] == .../New-ns3-integration
#     # parents[3] == .../Federated
#     candidates = [
#         here.parents[1] / "ns3-fl-network",  # <== inside ns-fl (your layout)
#         here.parents[1] / "ns3",
#         here.parents[2] / "ns3-fl-network",
#         here.parents[2] / "ns3",
#         here.parents[3] / "ns3-fl-network",
#         here.parents[3] / "ns3",
#     ]
#     for c in candidates:
#         if (c / "ns3").exists() or (c / "contrib").exists() or (c / "build").exists() or (c / "CMakeLists.txt").exists():
#             return c

#     tried = [str(x) for x in candidates]
#     raise FileNotFoundError(
#         "Could not locate your ns-3 tree.\n"
#         "Set network.ns3_root in config or NS3_ROOT env var.\n"
#         "Tried:\n  - " + "\n  - ".join(tried)
#     )

# # ------------------------ Adapter ------------------------

# class Network:
#     def __init__(self, cfg_any: Any):
#         cfg = _as_dict(cfg_any)
#         self.cfg = cfg
#         net = cfg.get("network", {}) or {}
#         self.network_type = str(net.get("type", "wifi")).lower()
#         self.controller   = str(net.get("controller", "socket")).lower()
#         self.num_clients  = int((cfg.get("clients") or {}).get("total", 1))
#         self.per_round    = int((cfg.get("clients") or {}).get("per_round", 1))

#         model_cfg = cfg.get("model", {}) or {}
#         self.model_size_kb = int(model_cfg.get("size", 0))               # KB
#         self.model_bytes   = self.model_size_kb * 1024                   # bytes (THz uses bytes)

#         sock = net.get("socket", {}) or {}
#         self.host   = str(sock.get("host", "127.0.0.1"))
#         self.port   = int(sock.get("port", 8080))
#         self.launch = bool(sock.get("launch", True))

#         self.wifi_cfg = net.get("wifi", {}) or {}
#         self.eth_cfg  = net.get("ethernet", {}) or {}
#         self.thz_cfg  = net.get("thz", {}) or {}

#         self._use_thz = (self.network_type == "thz")
#         self._sock: Optional[socket.socket] = None
#         self._child: Optional[subprocess.Popen] = None

#         # Locate ns-3 (terasim layout) and the ./ns3 helper
#         self.NS3_ROOT = _resolve_ns3_root(cfg)
#         self.NS3_CLI  = str(self.NS3_ROOT / "ns3")   # terasim helper

#         self._pending_sel: Optional[List[int]] = None

#         mode = "THz(JSON inline)" if self._use_thz else "Wi-Fi/Ethernet(socket)"
#         _log(f"initialized | mode={mode} | host={self.host} port={self.port} | ns3_root={self.NS3_ROOT}")

#         # We don’t blindly rebuild each run; build lazily if needed
#         if not self._use_thz and self.launch:
#             self._launch_socket_child()


#     # add this helper somewhere in the class (or as a staticmethod):
#     def _items_to_dict(self, items):
#         # items is a list of (eid, round_time, throughput)
#         return {int(eid): {"roundTime": float(rt), "throughput": float(thr)}
#                 for (eid, rt, thr) in items}



#     # ---------------- Lifecycle ----------------

#     def parse_clients(self, sample_clients):
#         """
#         Accepts a list of client objects or ints and returns a list of int IDs.
#         Compatible with asyncServer.py expectations.
#         """
#         return self._parse_ids(sample_clients)

#     def connect(self) -> bool:
#         if self._use_thz:
#             return True
#         if self._child is not None and self._child.poll() is not None:
#             _log(f"child exited rc={self._child.returncode}; relaunching")
#             self._launch_socket_child()
#         try:
#             if self._sock is None:
#                 self._connect_socket_with_retries((self.host, self.port), retries=80, delay=0.125)
#         except Exception as e:
#             tail = ""
#             try:
#                 if getattr(self, "_child_log", None):
#                     self._child_log.flush()
#                     with open(self._child_log.name, "r") as f:
#                         tail = "".join(f.readlines()[-100:])
#             except Exception:
#                 pass
#             rc = self._child.poll() if self._child else None
#             raise RuntimeError(f"[ns3] connect failed (rc={rc}): {e}\n--- wifi_exp.log tail ---\n{tail}\n---")

#         _log(f"connected | child={'up' if self._child and self._child.poll() is None else 'down'} sock={bool(self._sock)}")
#         return True

#     def disconnect(self) -> None:
#         try:
#             if self._sock:
#                 # politely tell provider to stop if supported
#                 try: self._sock.sendall(struct.pack("<II", CMD_EXIT, 0))
#                 except Exception: pass
#                 try: self._sock.shutdown(socket.SHUT_RDWR)
#                 except Exception: pass
#                 self._sock.close()
#         finally:
#             self._sock = None

#         if self._child is not None:
#             try:
#                 if self._child.poll() is None:
#                     _log("terminating child...")
#                     self._child.terminate()
#                     try: self._child.wait(timeout=3.0)
#                     except Exception:
#                         _log("killing child...")
#                         try: self._child.kill()
#                         except Exception: pass
#             finally:
#                 self._child = None
#         _log("disconnected")

#     # ---------------- Server API ----------------

#     def sendRequest(self, requestType: int, array: Iterable[int]) -> List[Tuple[int, float, float]]:
#         active_ids = self._parse_ids(array)
#         if not active_ids:
#             return []

#         if self._use_thz:
#             return self._run_thz(active_ids)

#         if self._sock is None:
#             self.connect()
#         if requestType != CMD_RUN_SIMULATION:
#             raise ValueError(f"unsupported requestType={requestType}")

#         # Request: header + compact list of selected client IDs
#         # Header: nItems must equal total number of clients on the C++ side
#         n = int(self.num_clients)
#         hdr = struct.pack("<II", CMD_RUN_SIMULATION, n)

#         # Body: n * <I> flags (1 if client selected this round, else 0)
#         sel = set(int(x) for x in active_ids)
#         flags = [1 if i in sel else 0 for i in range(n)]
#         body = struct.pack("<" + "I"*n, *flags)

#         self._send_all(hdr + body)

#         # Response: header + n × <Qdd>
#         items = self._recv_response(timeout=180.0)
#         return self._items_to_dict([t for t in items if int(t[0]) in sel])

#         # return [t for t in items if int(t[0]) in sel]

#     def sendAsyncRequest(self, requestType: int, array):
#         # Send only; do not block waiting for response
#         active_ids = self._parse_ids(array)
#         if not active_ids:
#             self._pending_sel = []
#             return

#         if self._use_thz:
#             # Defer running THz until readAsyncResponse
#             self._pending_sel = active_ids
#             return

#         if self._sock is None:
#             self.connect()
#         if requestType != CMD_RUN_SIMULATION:
#             raise ValueError(f"unsupported requestType={requestType}")

#         n = int(self.num_clients)
#         hdr = struct.pack("<II", CMD_RUN_SIMULATION, n)

#         sel = set(int(x) for x in active_ids)
#         flags = [1 if i in sel else 0 for i in range(n)]
#         body = struct.pack("<" + "I"*n, *flags)

#         self._send_all(hdr + body)
#         # remember selection so we can filter when the response arrives
#         self._pending_sel = list(active_ids)

#     def readAsyncResponse(self, timeout: float = 180.0):
#         sel = set(int(x) for x in (self._pending_sel or []))

#         if self._use_thz:
#             if not sel:
#                 return []
#             items = self._run_thz(list(sel))
#             self._pending_sel = None
#             return items

#         # socket path
#         items = self._recv_response(timeout=timeout)  # [(eid, rt, thr), ...]
#         self._pending_sel = None
#         return [t for t in items if (not sel) or int(t[0]) in sel]

#     # if you have readAsyncResponse/readSyncResponse, make them return the same dict:
#     def readSyncResponse(self, timeout=120.0):
#         return self._items_to_dict(self._recv_response(timeout=timeout))

#     def readAsyncResponse(self, timeout=120.0):
#         return self._items_to_dict(self._recv_response(timeout=timeout))

#     # ---------------- THz (JSON via ./ns3 run) ----------------

#     def _run_thz(self, active_ids: List[int]) -> List[Tuple[int, float, float]]:
#         # Build argv for scratch/thz-macro-central (you can extend params below)
#         args = [
#             self.NS3_CLI, "run", "scratch/thz-macro-central", "--",
#             f"--nodeNum={self.num_clients}",
#             f"--clients={len(active_ids)}",
#             f"--modelBytes={self.model_bytes}",
#         ]
#         # Optional THz knobs from config
#         for k, fmt in [
#             ("carrier_ghz", "--CarrierFreqGHz={}"),
#             ("pkt_size",    "--pktSize={}"),
#             ("sim_time",    "--simTime={}"),
#             ("interval_us", "--intervalUs={}"),
#             ("way",         "--way={}"),
#             ("radius",      "--radius={}"),
#             ("beamwidth",   "--beamwidth={}"),
#             ("gain",        "--gain={}"),
#             ("ap_angle",    "--apAngle={}"),
#             ("sta_angle",   "--staAngle={}"),
#             ("useWhiteList","--useWhiteList={}"),
#         ]:
#             if k in self.thz_cfg:
#                 args.append(fmt.format(self.thz_cfg[k]))

#         _log("[thz] " + " ".join(shlex.quote(x) for x in args))
#         proc = subprocess.run(args, cwd=str(self.NS3_ROOT),
#                               text=True, capture_output=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f"[thz] ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

#         # Find last JSON line on stdout
#         last = None
#         for line in reversed(proc.stdout.splitlines()):
#             s = line.strip()
#             if s.startswith("{") and s.endswith("}"):
#                 last = s; break
#         if not last:
#             raise RuntimeError("[thz] no JSON summary found in ns-3 output.")
#         data = json.loads(last)

#         # Map local ids (0..len-1) back to selected ids
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         items: List[Tuple[int, float, float]] = []
#         for e in data.get("clientResults", []):
#             local = int(e.get("id", -1))
#             if local not in id_map: continue
#             real = int(id_map[local])
#             rt   = float(e.get("doneAt", 0.0))
#             rx   = float(e.get("rxBytes", 0.0))
#             thr  = (rx / rt) if rt > 0 else 0.0
#             items.append((real, rt, thr))
#         return items

#     # ---------------- Wi-Fi/Ethernet child (./ns3 run wifi_exp) ----------------
#     def _find_wifi_bin_path(ns3_root: pathlib.Path) -> Optional[pathlib.Path]:
#         # already present? if not, add this helper near your utils
#         for d in (
#             ns3_root / "build" / "bin",
#             ns3_root / "cmake-cache" / "build" / "bin",
#             ns3_root / "cmake-cache" / "bin",
#         ):
#             for p in d.glob("ns*-wifi_exp*"):
#                 if p.is_file() and os.access(p, os.X_OK):
#                     return p
#         return None


#     def _launch_socket_child(self) -> None:
#         # Prefer explicit override from config
#         bin_override = (self.wifi_cfg.get("binary") if self.network_type == "wifi"
#                         else self.eth_cfg.get("binary"))
#         wifi_bin = pathlib.Path(bin_override).resolve() if bin_override else _find_wifi_bin_path(self.NS3_ROOT)
#         if not wifi_bin or not wifi_bin.exists():
#             _log("[ns3] building (wifi_exp missing)...")
#             self._ns3_build()
#             wifi_bin = pathlib.Path(bin_override).resolve() if bin_override else _find_wifi_bin_path(self.NS3_ROOT)
#         if not wifi_bin or not wifi_bin.exists():
#             raise FileNotFoundError("wifi_exp executable not found after build.")

#         argv = [str(wifi_bin)] + self._wifi_args()
#         _log("launch(bin): " + " ".join(shlex.quote(x) for x in argv))

#         # --- ALWAYS define env before Popen ---
#         env = os.environ.copy()
#         sep = ":" if os.name != "nt" else ";"
#         # ns-3’s cmake layout
#         libdir = self.NS3_ROOT / "build" / "lib"
#         bindir = self.NS3_ROOT / "build" / "bin"
#         pydir  = self.NS3_ROOT / "build" / "bindings" / "python"

#         if bindir.exists():
#             env["PATH"] = str(bindir) + (sep + env["PATH"] if "PATH" in env and env["PATH"] else "")
#         if os.name != "nt" and libdir.exists():
#             env["LD_LIBRARY_PATH"] = str(libdir) + (sep + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env and env["LD_LIBRARY_PATH"] else "")
#         if pydir.exists():
#             env["PYTHONPATH"] = str(pydir) + (sep + env["PYTHONPATH"] if "PYTHONPATH" in env and env["PYTHONPATH"] else "")

#         # Don’t block here; just spawn and let connect() do the socket handshake
#         self._child = subprocess.Popen(
#             argv,
#             cwd=str(self.NS3_ROOT),
#             env=env,
#             stdout=subprocess.DEVNULL,   # or keep a logfile if you prefer
#             stderr=subprocess.DEVNULL,   # (you can switch these to a file to debug)
#             text=True,
#         )
#         log_path = pathlib.Path(self.NS3_ROOT) / "wifi_exp.launch.log"
#         self._child_log = open(log_path, "a+", buffering=1)
#         self._child = subprocess.Popen(
#             argv, cwd=str(self.NS3_ROOT), env=env,
#             stdout=self._child_log, stderr=self._child_log, text=True
#         )
#         _log(f"child started (pid={self._child.pid}); logging → {log_path}")
#         time.sleep(0.3)  # let it bind


#     def _ns3_build(self) -> None:
#         if not self.NS3_ROOT.exists():
#             raise FileNotFoundError(f"ns-3 tree not found at {self.NS3_ROOT}")
#         proc = subprocess.run(self.NS3_CLI + " build", cwd=str(self.NS3_ROOT),
#                               shell=True, text=True,
#                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if proc.returncode != 0:
#             raise RuntimeError(f"ns-3 build failed:\n{proc.stderr}")
    
#     def _wifi_args(self) -> List[str]:
#         nic = self.wifi_cfg if self.network_type == "wifi" else self.eth_cfg
#         args = [
#             f"--NumClients={self.num_clients}",
#             f"--NetworkType={'wifi' if self.network_type == 'wifi' else 'ethernet'}",
#             f"--MaxPacketSize={int(nic.get('max_packet_size', 1024))}",
#             f"--TxGain={float(nic.get('tx_gain', 0.0))}",
#             f"--ModelSize={int(self.model_size_kb)}",   # KB
#             "--LearningModel=sync",
#             "--UseFLSocket=true",                       # REQUIRED in this fork
#         ]
#         if "data_rate" in nic:
#             args.append(f"--DataRate={str(nic['data_rate'])}")
#         if "sim_time" in nic:
#             args.append(f"--SimTime={float(nic['sim_time'])}")
#         return args




#     # def _wifi_args(self) -> List[str]:
#     #     nic = self.wifi_cfg if self.network_type == "wifi" else self.eth_cfg
#     #     args = [
#     #         "--UseFLSocket=true",
#     #         f"--NumClients={self.num_clients}",
#     #         f"--ModelSize={int(self.model_size_kb)}",   # KB for wifi_exp
#     #         "--LearningModel=sync",                     # ensure C++ emits responses
#     #         f"--Port={self.port}",
#     #         f"--Host={self.host}",
#     #     ]
#     #     if "max_packet_size" in nic: args += [f"--MaxPacketSize={int(nic['max_packet_size'])}"]
#     #     if "data_rate" in nic:       args += [f"--DataRate={str(nic['data_rate'])}"]
#     #     if "tx_gain" in nic:         args += [f"--TxGain={float(nic['tx_gain'])}"]
#     #     if "sim_time" in nic:        args += [f"--SimTime={float(nic['sim_time'])}"]
#     #     return args

#     # ---------------- Socket helpers ----------------

#     def _send_all(self, payload: bytes) -> None:
#         try:
#             self._sock.sendall(payload)  # type: ignore[union-attr]
#         except (BrokenPipeError, ConnectionResetError):
#             _log("send failed; attempting reconnect")
#             self._connect_socket_with_retries((self.host, self.port), retries=40, delay=0.25)
#             self._sock.sendall(payload)  # type: ignore[union-attr]

#     def _recv_all(self, n: int, timeout: float = 60.0) -> bytes:
#         if self._sock is None:
#             raise RuntimeError("[NET][socket] not connected")
#         self._sock.settimeout(timeout)
#         buf = bytearray()
#         while len(buf) < n:
#             chunk = self._sock.recv(n - len(buf))
#             if not chunk:
#                 rc = self._child.returncode if (self._child and self._child.poll() is not None) else None
#                 tail = ""
#                 if self._child and self._child.stderr:
#                     try:
#                         err = self._child.stderr.read()
#                         if err: tail = err[-4096:]
#                     except Exception:
#                         pass
#                 raise RuntimeError(f"[NET][socket] connection closed (child rc={rc}). ns-3 stderr tail:\n{tail}")
#             buf.extend(chunk)
#         return bytes(buf)

#     def _recv_response(self, timeout: float) -> List[Tuple[int, float, float]]:
#         hdr = self._recv_all(8, timeout=timeout)
#         (cmd, n) = struct.unpack("<II", hdr)
#         if cmd != CMD_RESPONSE:
#             raise RuntimeError(f"[NET][socket] unexpected cmd={cmd}, expected RESPONSE=0")
#         out: List[Tuple[int, float, float]] = []
#         for _ in range(n):
#             blob = self._recv_all(24, timeout=timeout)  # Qdd
#             (eid, rt, thr) = struct.unpack("<Qdd", blob)
#             out.append((int(eid), float(rt), float(thr)))
#         return out

#     def _connect_socket_with_retries(self, addr: Tuple[str, int], retries: int, delay: float):
#         (host, port) = addr
#         last_err: Optional[BaseException] = None
#         for i in range(retries):
#             try:
#                 s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
#                 s.connect((host, port))
#                 self._sock = s
#                 _log(f"socket connected to {host}:{port} (attempt {i+1}/{retries})")
#                 return
#             except Exception as e:
#                 last_err = e
#                 time.sleep(delay)
#         raise ConnectionError(f"could not connect to {host}:{port} after {retries} attempts: {last_err}")

#     # ---------------- Small helpers ----------------

#     @staticmethod
#     def _parse_ids(arr: Iterable[Any]) -> List[int]:
#         # supports list of ints or objects with .client_id
#         ids: List[int] = []
#         for x in arr:
#             if hasattr(x, "client_id"):
#                 ids.append(int(getattr(x, "client_id")))
#             else:
#                 ids.append(int(x))
#         return ids




















# # flsim/network.py — ns-3 THz runner + Wi-Fi/Ethernet
# # Modes:
# #  - Inline (default): launch ns-3 for each request and parse JSON/fallback.
# #  - Socket controller: keep one ns-3 "wifi_exp" running and speak its binary protocol.
# import json
# import socket
# import struct
# import subprocess
# import time
# from typing import Any, Dict, List, Optional, Tuple

# PATH = '../ns3-fl-network'
# THZ_PROGRAM = 'scratch/thz-macro-central'
# WIFI_PROGRAM = 'wifi_exp'

# def _get(root: Any, path: List[str], default=None):
#     cur = root
#     for k in path:
#         if isinstance(cur, dict) and k in cur:
#             cur = cur[k]
#         elif hasattr(cur, k):
#             cur = getattr(cur, k)
#         else:
#             return default
#     return cur

# def _parse_rate_to_Bps(s: str, default_Bps: float = (250_000 / 8)) -> float:
#     try:
#         s = (s or '').strip().lower()
#         if not s:
#             return default_Bps
#         num = ''
#         unit = ''
#         for ch in s:
#             if ch.isdigit() or ch in '.+-eE':
#                 num += ch
#             else:
#                 unit += ch
#         val = float(num)
#         unit = unit.strip()
#         scale = 1.0
#         if unit in ('bps', ''):
#             scale = 1.0
#         elif unit == 'kbps':
#             scale = 1e3
#         elif unit == 'mbps':
#             scale = 1e6
#         elif unit == 'gbps':
#             scale = 1e9
#         else:
#             scale = 1.0
#         return (val * scale) / 8.0
#     except Exception:
#         return default_Bps

# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(_get(self.config, ['clients', 'total'], 1))

#         # Backend selector
#         self.network_type = str(_get(self.config, ['network', 'type'], 'thz')).lower()
#         self._use_thz = self.network_type not in ('wifi', 'ethernet')

#         # Controller selector: 'inline' (default) or 'socket'
#         self.controller = str(_get(self.config, ['network', 'controller'], 'inline')).lower()
#         sock_cfg = _get(self.config, ['network', 'socket'], {}) or {}
#         self._sock_host = str(sock_cfg.get('host', '127.0.0.1'))
#         self._sock_port = int(sock_cfg.get('port', 8080))
#         self._sock_launch = bool(sock_cfg.get('launch', False))

#         # THz knobs
#         thz = _get(self.config, ['network', 'thz'], {}) or {}
#         self._thz_cfg = {
#             'carrier_ghz':  float(thz.get('carrier_ghz', 300.0)),
#             'pkt_size':     int(thz.get('pkt_size', 600)),
#             'sim_time':     float(thz.get('sim_time', 0.8)),
#             'interval_us':  int(thz.get('interval_us', 20)),
#             'way':          int(thz.get('way', 3)),
#             'radius':       float(thz.get('radius', 0.5)),
#             'beamwidth':    float(thz.get('beamwidth', 40)),
#             'gain':         float(thz.get('gain', 30)),
#             'ap_angle':     float(thz.get('ap_angle', 0)),
#             'sta_angle':    float(thz.get('sta_angle', 180)),
#             'useWhiteList': int(thz.get('useWhiteList', 0)),
#         }

#         # Wi-Fi / Ethernet knobs
#         self._wifi_cfg = _get(self.config, ['network', 'wifi'], {}) or {}
#         self._eth_cfg = _get(self.config, ['network', 'ethernet'], {}) or {}

#         # Model bytes per upload — config expresses KB, convert to bytes
#         self._model_bytes = int(_get(self.config, ['model', 'size'], 1600)) * 1024

#         # Build ns-3 once
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # Async / process state
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_queue: List[Dict[int, Dict[str, float]]] = []
#         self._deadline: Optional[float] = None

#         # Socket state
#         self._s: Optional[socket.socket] = None
#         self._ns3_proc: Optional[subprocess.Popen] = None

#     # --------------------------------------------------------------
#     def connect(self):
#         return

#     def disconnect(self):
#         try:
#             if self._s:
#                 try:
#                     # Send EXIT/END if supported
#                     self._s.send(struct.pack("II", 3, 0))
#                 except Exception:
#                     pass
#                 self._s.close()
#         finally:
#             self._s = None
#         if self._ns3_proc:
#             try:
#                 self._ns3_proc.terminate()
#             except Exception:
#                 pass
#             self._ns3_proc = None

#     def parse_clients(self, clients):
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     # --------------------------------------------------------------
#     def _thz_cmd(self, *, total_clients: int, active_count: int, model_bytes: int) -> List[str]:
#         t = self._thz_cfg
#         return [
#             './ns3', 'run', THZ_PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--CarrierFreqGHz={t["carrier_ghz"]}',
#             f'--pktSize={t["pkt_size"]}',
#             f'--simTime={t["sim_time"]}',
#             f'--intervalUs={t["interval_us"]}',
#             f'--way={t["way"]}',
#             f'--radius={t["radius"]}',
#             f'--beamwidth={t["beamwidth"]}',
#             f'--gain={t["gain"]}',
#             f'--apAngle={t["ap_angle"]}',
#             f'--staAngle={t["sta_angle"]}',
#             f'--useWhiteList={t["useWhiteList"]}',
#         ]

#     def _inline_net_cmd(self, *, active_count: int, model_bytes: int) -> Tuple[List[str], float]:
#         if self.network_type == 'wifi':
#             ncfg = self._wifi_cfg
#         else:
#             ncfg = self._eth_cfg
#         max_pkt   = int(ncfg.get('max_packet_size', 1024))
#         data_rate = str(ncfg.get('data_rate', '250kbps'))
#         tx_gain   = float(ncfg.get('tx_gain', 0.0)) if self.network_type == 'wifi' else None
#         learning  = str(_get(self.config, ['server'], 'sync'))
#         args = [
#             f'--NumClients={active_count}',
#             f'--NetworkType={self.network_type}',
#             f'--ModelSize={model_bytes}',
#             f'--MaxPacketSize={max_pkt}',
#             f'--DataRate={data_rate}',
#             f'--LearningModel={learning}',
#         ]
#         if tx_gain is not None:
#             args.append(f'--TxGain={tx_gain}')
#         cmd = ['./ns3', 'run', WIFI_PROGRAM, '--', *args]
#         exp_Bps = _parse_rate_to_Bps(data_rate)
#         return cmd, exp_Bps

#     # Socket controller helpers
#     def _ensure_socket_up(self):
#         """Launch ns-3 wifi_exp in socket mode (once) and connect."""
#         if self._s and self._ns3_proc and self._ns3_proc.poll() is None:
#             return

#         # Launch ns-3 (optional)
#         if self._sock_launch and (self._ns3_proc is None or self._ns3_proc.poll() is not None):
#             if self.network_type == 'wifi':
#                 ncfg = self._wifi_cfg
#             else:
#                 ncfg = self._eth_cfg
#             max_pkt   = int(ncfg.get('max_packet_size', 1024))
#             data_rate = str(ncfg.get('data_rate', '250kbps'))
#             tx_gain   = float(ncfg.get('tx_gain', 0.0)) if self.network_type == 'wifi' else None
#             learning  = str(_get(self.config, ['server'], 'sync'))

#             exp_Bps = _parse_rate_to_Bps(data_rate, default_Bps=250000/8)
#             est_rt = max(1.0, self._model_bytes / max(1.0, exp_Bps))
#             sim_time = max(10.0, 8.0 * est_rt)

#             args = [
#                 './ns3', 'run', WIFI_PROGRAM, '--',
#                 f'--UseFLSocket=true',
#                 f'--NumClients={self.num_clients}',
#                 f'--NetworkType={self.network_type}',
#                 f'--ModelSize={int(self._model_bytes/1000)}',  # KB -> bytes in C++ via *1000
#                 f'--MaxPacketSize={max_pkt}',
#                 f'--DataRate={data_rate}',
#                 f'--LearningModel={learning}',
#                 f'--SimTime={sim_time:.3f}',
#             ]
#             if tx_gain is not None:
#                 args.append(f'--TxGain={tx_gain}')

#             self._ns3_proc = subprocess.Popen(
#                 args, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )
#             time.sleep(0.2)

#         # Connect
#         self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self._s.settimeout(0.5)
#         last_err = None
#         for _ in range(20):
#             try:
#                 self._s.connect((self._sock_host, self._sock_port))
#                 self._s.settimeout(0.0)  # non-blocking for polling
#                 return
#             except Exception as e:
#                 last_err = e
#                 time.sleep(0.1)
#         raise RuntimeError(f'Could not connect to ns-3 controller at {self._sock_host}:{self._sock_port}: {last_err}')

#     @staticmethod
#     def _parse_last_json(stdout: str) -> Dict[str, Any]:
#         last = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     @staticmethod
#     def _extract_times(entry: Dict[str, Any], default_sim_time: float) -> float:
#         if 'doneAt' in entry:
#             return float(entry['doneAt'])
#         if 'endTime' in entry:
#             return float(entry['endTime'])
#         if 'roundTime' in entry:
#             return float(entry['roundTime'])
#         return default_sim_time

#     # --------------------------------------------------------------
#     # SYNC API
#     def sendRequest(self, *, requestType: int, array: list):
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)
#         if not active_ids:
#             return {}

#         # Socket controller path
#         if not self._use_thz and self.controller == 'socket':
#             self._ensure_socket_up()
#             # [int32 command, int32 n] + n × [int32 id]
#             try:
#                 self._s.send(struct.pack("II", int(requestType), len(active_ids)))
#                 for ele in active_ids:
#                     self._s.send(struct.pack("I", int(ele)))
#             except (BrokenPipeError, OSError):
#                 self.disconnect()
#                 self._ensure_socket_up()
#                 self._s.send(struct.pack("II", int(requestType), len(active_ids)))
#                 for ele in active_ids:
#                     self._s.send(struct.pack("I", int(ele)))

#             # Blocking read for short sync response
#             self._s.settimeout(5.0)
#             hdr = self._s.recv(8)
#             if len(hdr) < 8:
#                 raise RuntimeError(f'ns-3 socket: short sync header ({len(hdr)} bytes)')
#             command, nItems = struct.unpack("II", hdr)

#             out: Dict[int, Dict[str, float]] = {}
#             for _ in range(nItems):
#                 body = self._s.recv(8 * 3)  # id(uint64), roundTime(double), throughput(double)
#                 if len(body) < 8 * 3:
#                     raise RuntimeError(f'ns-3 socket: short sync body ({len(body)} bytes)')
#                 eid, roundTime, throughput = struct.unpack("Qdd", body)
#                 out[int(eid)] = {"roundTime": float(roundTime), "throughput": float(throughput)}
#             self._s.settimeout(0.0)
#             return out

#         # THz path (unchanged)
#         if self._use_thz:
#             cmd = self._thz_cmd(
#                 total_clients=self.num_clients,
#                 active_count=len(active_ids),
#                 model_bytes=self._model_bytes,
#             )
#             proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             if proc.returncode != 0:
#                 raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
#             data = self._parse_last_json(proc.stdout)
#             id_map = {local: active_ids[local] for local in range(len(active_ids))}
#             out = {}
#             for e in data.get('clientResults', []):
#                 local = int(e.get('id', -1))
#                 if local not in id_map:
#                     continue
#                 rx_bytes = float(e.get('rxBytes', 0.0))
#                 done_at  = self._extract_times(e, self._thz_cfg['sim_time'])
#                 thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#                 out[id_map[local]] = {'roundTime': done_at, 'throughput': thr}
#             return out

#         # Inline Wi-Fi/Ethernet fallback (no controller)
#         cmd, exp_Bps = self._inline_net_cmd(active_count=len(active_ids), model_bytes=self._model_bytes)
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
#         out: Dict[int, Dict[str, float]] = {}
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         try:
#             data = self._parse_last_json(proc.stdout)
#             for e in data.get('clientResults', []):
#                 local = int(e.get('id', -1))
#                 if local not in id_map:
#                     continue
#                 rx_bytes = float(e.get('rxBytes', 0.0))
#                 rt = self._extract_times(e, 0.0)
#                 if not rt or rt <= 0.0:
#                     rt = (self._model_bytes / max(1.0, exp_Bps))
#                 thr = (rx_bytes / rt) if rt > 0 else 0.0
#                 out[id_map[local]] = {'roundTime': rt, 'throughput': thr}
#             if out:
#                 return out
#         except Exception:
#             pass
#         est_rt = (self._model_bytes / max(1.0, exp_Bps))
#         est_thr = exp_Bps
#         for real_id in active_ids:
#             out[real_id] = {'roundTime': est_rt, 'throughput': est_thr}
#         return out

#     # --------------------------------------------------------------
#     # ASYNC API
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         self._async_ids = active_ids
#         self._async_queue = []
#         self._deadline = None

#         if not active_ids:
#             self._proc = None
#             self._async_queue = []
#             return

#         # Socket controller path
#         if not self._use_thz and self.controller == 'socket':
#             self._ensure_socket_up()
#             try:
#                 self._s.send(struct.pack("II", int(requestType), len(active_ids)))
#                 for ele in active_ids:
#                     self._s.send(struct.pack("I", int(ele)))
#             except (BrokenPipeError, OSError):
#                 self.disconnect()
#                 self._ensure_socket_up()
#                 self._s.send(struct.pack("II", int(requestType), len(active_ids)))
#                 for ele in active_ids:
#                     self._s.send(struct.pack("I", int(ele)))
#             self._s.settimeout(0.0)
#             ncfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#             exp_Bps = _parse_rate_to_Bps(str(ncfg.get('data_rate', '250kbps')))
#             est_rt = self._model_bytes / max(1.0, exp_Bps)
#             self._deadline = time.time() + max(15.0, 4.0 * est_rt)
#             return

#         if self._use_thz:
#             cmd = self._thz_cmd(
#                 total_clients=self.num_clients,
#                 active_count=len(active_ids),
#                 model_bytes=self._model_bytes,
#             )
#             self._proc = subprocess.Popen(
#                 cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )
#             sim_t = self._thz_cfg['sim_time']
#             self._deadline = time.time() + max(10.0, 4.0 * sim_t)
#             return

#         # Inline Wi-Fi/Ethernet
#         cmd, exp_Bps = self._inline_net_cmd(active_count=len(active_ids), model_bytes=self._model_bytes)
#         self._proc = subprocess.Popen(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         est_rt = self._model_bytes / max(1.0, exp_Bps)
#         self._deadline = time.time() + max(15.0, 4.0 * est_rt)

#     def _sock_recv_exact(self, n: int) -> Optional[bytes]:
#         """Non-blocking 'read exactly n bytes' helper for the socket path."""
#         if not self._s:
#             return None
#         chunks: List[bytes] = []
#         got = 0
#         for _ in range(64):
#             try:
#                 b = self._s.recv(n - got)
#                 if not b:
#                     return None
#                 chunks.append(b)
#                 got += len(b)
#                 if got >= n:
#                     return b''.join(chunks)
#             except (BlockingIOError, socket.timeout):
#                 break
#             except OSError:
#                 return None
#         return None

#     def readAsyncResponse(self):
#         # Nothing started (inline path)
#         if self._proc is None and self.controller != 'socket' and not self._async_queue:
#             return 'end'

#         # Socket controller path
#         if not self._use_thz and self.controller == 'socket':
#             hdr = self._sock_recv_exact(8)
#             if hdr is None:
#                 if self._deadline and time.time() > self._deadline:
#                     ncfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#                     exp_Bps = _parse_rate_to_Bps(str(ncfg.get('data_rate', '250kbps')))
#                     est_rt = self._model_bytes / max(1.0, exp_Bps)
#                     for real_id in self._async_ids:
#                         self._async_queue.append({
#                             real_id: {'startTime': 0.0, 'endTime': est_rt, 'throughput': exp_Bps}
#                         })
#                 else:
#                     return {}
#             else:
#                 command, nItems = struct.unpack("II", hdr)
#                 if command == 3:
#                     return 'end'
#                 out_ordered: List[Dict[int, Dict[str, float]]] = []
#                 for _ in range(nItems):
#                     body = self._sock_recv_exact(8 * 4)
#                     if body is None:
#                         return {}
#                     eid, startTime, endTime, throughput = struct.unpack("Qddd", body)
#                     out_ordered.append({
#                         int(eid): {
#                             'startTime': float(startTime),
#                             'endTime': float(endTime),
#                             'throughput': float(throughput)
#                         }
#                     })
#                 self._async_queue.extend(out_ordered)

#             if self._async_queue:
#                 return self._async_queue.pop(0)
#             return 'end'

#         # Inline processes path (THz or Wi-Fi/Ethernet)
#         import subprocess as sp
#         if self._proc is not None and self._proc.poll() is None:
#             if self._deadline is not None and time.time() > self._deadline:
#                 try:
#                     self._proc.terminate()
#                 except Exception:
#                     pass
#                 try:
#                     stdout, stderr = self._proc.communicate(timeout=2)
#                 except sp.TimeoutExpired:
#                     try:
#                         self._proc.kill()
#                     except Exception:
#                         pass
#                     stdout, stderr = self._proc.communicate()
#                 finally:
#                     self._proc = None

#                 data = {}
#                 try:
#                     data = self._parse_last_json(stdout)
#                 except Exception:
#                     data = {'clientResults': []}

#                 id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#                 results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

#                 exp_Bps = None
#                 if not self._use_thz:
#                     cfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#                     exp_Bps = _parse_rate_to_Bps(str(cfg.get('data_rate', '250kbps')))

#                 for local in range(len(self._async_ids)):
#                     ent = results.get(local)
#                     real_id = id_map[local]
#                     if ent:
#                         default_t = self._thz_cfg['sim_time'] if self._use_thz else (self._model_bytes / max(1.0, (exp_Bps or 1.0)))
#                         rx_bytes = float(ent.get('rxBytes', 0.0))
#                         done_at  = self._extract_times(ent, default_t)
#                         thr = (rx_bytes / done_at) if done_at and done_at > 0 else (exp_Bps or 0.0)
#                     else:
#                         if self._use_thz:
#                             done_at = self._thz_cfg['sim_time']; thr = 0.0
#                         else:
#                             done_at = (self._model_bytes / max(1.0, (exp_Bps or 1.0))); thr = (exp_Bps or 0.0)
#                     self._async_queue.append({real_id: {'startTime': 0.0, 'endTime': done_at, 'throughput': thr}})
#             else:
#                 return {}

#         if self._proc is not None and self._proc.poll() is not None:
#             stdout, stderr = self._proc.communicate()
#             self._proc = None

#             data = {}
#             parsed = False
#             try:
#                 data = self._parse_last_json(stdout)
#                 parsed = True
#             except Exception:
#                 parsed = False

#             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#             results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])} if parsed else {}

#             exp_Bps = None
#             if not self._use_thz:
#                 cfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#                 exp_Bps = _parse_rate_to_Bps(str(cfg.get('data_rate', '250kbps')))
#                 est_rt = self._model_bytes / max(1.0, exp_Bps)

#             for local in range(len(self._async_ids)):
#                 real_id = id_map[local]
#                 ent = results.get(local)
#                 if ent:
#                     rx_bytes = float(ent.get('rxBytes', 0.0))
#                     done_at  = self._extract_times(ent, self._thz_cfg['sim_time'] if self._use_thz else (self._model_bytes / max(1.0, (exp_Bps or 1.0))))
#                     thr = (rx_bytes / done_at) if done_at and done_at > 0 else (exp_Bps or 0.0)
#                 else:
#                     if self._use_thz:
#                         done_at = self._thz_cfg['sim_time']; thr = 0.0
#                     else:
#                         done_at = est_rt; thr = exp_Bps or 0.0
#                 self._async_queue.append({real_id: {'startTime': 0.0, 'endTime': done_at, 'throughput': thr}})

#         if self._async_queue:
#             return self._async_queue.pop(0)
#         return 'end'














































# # flsim/network.py — ns-3 THz runner + Wi-Fi/Ethernet
# # modes: (a) standalone per-round, (b) socket controller (upstream style)
# import json
# import socket
# import struct
# import subprocess
# import time
# from typing import Any, Dict, List, Optional, Tuple

# PATH = '../ns3-fl-network'
# THZ_PROGRAM = 'scratch/thz-macro-central'
# WIFI_PROGRAM = 'wifi_exp'


# def _get(root: Any, path: List[str], default=None):
#     cur = root
#     for k in path:
#         if isinstance(cur, dict) and k in cur:
#             cur = cur[k]
#         elif hasattr(cur, k):
#             cur = getattr(cur, k)
#         else:
#             return default
#     return cur


# def _parse_rate_to_Bps(s: str, default_Bps: float = (250_000 / 8)) -> float:
#     """Parse strings like '250kbps', '6Mbps', '1Gbps' into BYTES/second."""
#     try:
#         s = (s or '').strip().lower()
#         if not s:
#             return default_Bps
#         num = ''
#         unit = ''
#         for ch in s:
#             if ch.isdigit() or ch in '.+-eE':
#                 num += ch
#             else:
#                 unit += ch
#         val = float(num)
#         unit = unit.strip()
#         if unit in ('bps', ''):
#             scale = 1.0
#         elif unit == 'kbps':
#             scale = 1e3
#         elif unit == 'mbps':
#             scale = 1e6
#         elif unit == 'gbps':
#             scale = 1e9
#         else:
#             scale = 1.0
#         return (val * scale) / 8.0  # bits/s -> bytes/s
#     except Exception:
#         return default_Bps


# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(_get(self.config, ['clients', 'total'], 1))

#         # Backend selector
#         self.network_type = str(_get(self.config, ['network', 'type'], 'thz')).lower()
#         self.controller = str(_get(self.config, ['network', 'controller'], 'standalone')).lower()
#         self._use_thz = self.network_type not in ('wifi', 'ethernet')

#         # THz knobs
#         thz = _get(self.config, ['network', 'thz'], {}) or {}
#         self._thz_cfg = {
#             'carrier_ghz':  float(thz.get('carrier_ghz', 300.0)),
#             'pkt_size':     int(thz.get('pkt_size', 600)),
#             'sim_time':     float(thz.get('sim_time', 0.8)),
#             'interval_us':  int(thz.get('interval_us', 20)),
#             'way':          int(thz.get('way', 3)),
#             'radius':       float(thz.get('radius', 0.5)),
#             'beamwidth':    float(thz.get('beamwidth', 40)),
#             'gain':         float(thz.get('gain', 30)),
#             'ap_angle':     float(thz.get('ap_angle', 0)),
#             'sta_angle':    float(thz.get('sta_angle', 180)),
#             'useWhiteList': int(thz.get('useWhiteList', 0)),
#         }

#         # Wi-Fi / Ethernet knobs
#         self._wifi_cfg = _get(self.config, ['network', 'wifi'], {}) or {}
#         self._eth_cfg  = _get(self.config, ['network', 'ethernet'], {}) or {}
#         self._sock_cfg = _get(self.config, ['network', 'socket'], {}) or {}
#         self._sock_host = str(self._sock_cfg.get('host', '127.0.0.1'))
#         self._sock_port = int(self._sock_cfg.get('port', 8080))
#         self._sock_launch = bool(self._sock_cfg.get('launch', True))

#         # model.size is in KB in your configs → convert to bytes (canonical)
#         self._model_bytes = int(_get(self.config, ['model', 'size'], 1600)) * 1024

#         # Build ns-3 once
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # async / process state
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_queue: List[Dict[int, Dict[str, float]]] = []
#         self._deadline: Optional[float] = None

#         # socket controller state
#         self._sock: Optional[socket.socket] = None
#         self._sock_proc: Optional[subprocess.Popen] = None

#         if (not self._use_thz) and self.controller == 'socket':
#             # start controller backend once and connect
#             self._ensure_socket_backend_started()

#     # ------------------------------------------------------------------
#     # compatibility no-ops (old TCP control plane)
#     def connect(self): 
#         return

#     def disconnect(self):
#         # Tear down socket controller if we started it
#         if self._sock is not None:
#             try:
#                 # COMMAND::EXIT (3) with 0 items (matches upstream)
#                 self._sock.sendall(struct.pack('II', 3, 0))
#             except Exception:
#                 pass
#             try:
#                 self._sock.close()
#             except Exception:
#                 pass
#             self._sock = None
#         if self._sock_proc is not None:
#             try:
#                 self._sock_proc.terminate()
#             except Exception:
#                 pass
#             self._sock_proc = None

#     # accept list of client objects or raw ids
#     def parse_clients(self, clients):
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     # ------------------------------------------------------------------
#     # core ns-3 launchers
#     def _thz_cmd(self, *, total_clients: int, active_count: int, model_bytes: int) -> List[str]:
#         t = self._thz_cfg
#         return [
#             './ns3', 'run', THZ_PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--CarrierFreqGHz={t["carrier_ghz"]}',
#             f'--pktSize={t["pkt_size"]}',
#             f'--simTime={t["sim_time"]}',
#             f'--intervalUs={t["interval_us"]}',
#             f'--way={t["way"]}',
#             f'--radius={t["radius"]}',
#             f'--beamwidth={t["beamwidth"]}',
#             f'--gain={t["gain"]}',
#             f'--apAngle={t["ap_angle"]}',
#             f'--staAngle={t["sta_angle"]}',
#             f'--useWhiteList={t["useWhiteList"]}',
#         ]

#     def _wifi_cmd(self, *, active_count: int, model_bytes: int, use_socket: bool) -> Tuple[List[str], float]:
#         """Build a cmdline for wifi_exp/ethernet and return (cmd, expected_Bps)."""
#         ncfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#         max_pkt   = int(ncfg.get('max_packet_size', 1024))
#         data_rate = str(ncfg.get('data_rate', '250kbps'))
#         tx_gain   = float(ncfg.get('tx_gain', 0.0)) if self.network_type == 'wifi' else None
#         learning  = str(_get(self.config, ['server'], 'sync'))

#         args = [
#             f'--NumClients={self.num_clients}',    # total pool when using controller
#             f'--NetworkType={self.network_type}',
#             f'--MaxPacketSize={max_pkt}',
#             f'--DataRate={data_rate}',
#             f'--LearningModel={learning}',
#         ]
#         if tx_gain is not None:
#             args.append(f'--TxGain={tx_gain}')

#         # ModelSize: wifi_exp expects *KB* and multiplies by 1000 internally.
#         model_kb = max(1, int(round(model_bytes / 1000.0)))
#         args.append(f'--ModelSize={model_kb}')

#         if use_socket:
#             args.append('--UseFLSocket=true')
#         # Optional SimTime keeps the sim from running forever in standalone
#         if not use_socket:
#             # rough upper bound so the sim ends even if no JSON
#             est_Bps = _parse_rate_to_Bps(data_rate)
#             est_rt = model_bytes / max(1.0, est_Bps)
#             sim_time = max(1.0, 4.0 * est_rt)
#             args.append(f'--SimTime={sim_time:.2f}')

#         cmd = ['./ns3', 'run', WIFI_PROGRAM, '--', *args]
#         exp_Bps = _parse_rate_to_Bps(data_rate)
#         return cmd, exp_Bps

#     @staticmethod
#     def _parse_last_json(stdout: str) -> Dict[str, Any]:
#         last = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     @staticmethod
#     def _extract_times(entry: Dict[str, Any], default_sim_time: float) -> float:
#         if 'doneAt' in entry:
#             return float(entry['doneAt'])
#         if 'endTime' in entry:
#             return float(entry['endTime'])
#         if 'roundTime' in entry:
#             return float(entry['roundTime'])
#         return default_sim_time

#     # ------------------------------------------------------------------
#     # Socket controller helpers (upstream protocol)
#     def _ensure_socket_backend_started(self):
#         # Launch wifi_exp with socket support?
#         if self._sock_launch and self._sock_proc is None:
#             cmd, _ = self._wifi_cmd(active_count=self.num_clients,
#                                     model_bytes=self._model_bytes,
#                                     use_socket=True)
#             self._sock_proc = subprocess.Popen(
#                 cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )

#         # Connect to the provider
#         deadline = time.time() + 10.0
#         last_err = None
#         while time.time() < deadline:
#             try:
#                 s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
#                 s.settimeout(2.0)
#                 s.connect((self._sock_host, self._sock_port))
#                 self._sock = s
#                 return
#             except Exception as e:
#                 last_err = e
#                 time.sleep(0.25)
#         raise RuntimeError(f'Could not connect to wifi_exp controller at '
#                            f'{self._sock_host}:{self._sock_port}: {last_err}')

#     def _socket_send_round(self, *, requestType: int, ids: List[int]) -> List[Dict[str, float]]:
#         """
#         Protocol:
#           send: header = struct('II') -> (cmd, n)
#                 then n times struct('I') -> client id (uint32)
#           recv: header = struct('II') -> (cmd, n)
#                 then n times struct('Qddd') -> (id:uint64, start, end, thr)
#         """
#         if self._sock is None:
#             self._ensure_socket_backend_started()

#         # Send header and ids
#         hdr = struct.pack('II', int(requestType), len(ids))
#         payload = b''.join(struct.pack('I', int(x)) for x in ids)
#         self._sock.sendall(hdr + payload)

#         # Receive header
#         header = self._recv_exact(8)
#         cmd, n_items = struct.unpack('II', header)

#         results: List[Dict[str, float]] = []
#         for _ in range(n_items):
#             blob = self._recv_exact(8 + 8 + 8 + 8)  # Q d d d
#             cid_u64, t0, t1, thr = struct.unpack('Qddd', blob)
#             cid = int(cid_u64 & 0xFFFFFFFFFFFFFFFF)
#             results.append({'id': cid, 'startTime': float(t0), 'endTime': float(t1), 'throughput': float(thr)})
#         return results

#     def _recv_exact(self, n: int) -> bytes:
#         buf = bytearray()
#         while len(buf) < n:
#             chunk = self._sock.recv(n - len(buf))
#             if not chunk:
#                 raise RuntimeError('socket closed')
#             buf.extend(chunk)
#         return bytes(buf)

#     # ------------------------------------------------------------------
#     # SYNC API
#     def sendRequest(self, *, requestType: int, array: list):
#         # bitmap or list of ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         if not active_ids:
#             return {}

#         # ---- THz path ----
#         if self._use_thz:
#             cmd = self._thz_cmd(
#                 total_clients=self.num_clients,
#                 active_count=len(active_ids),
#                 model_bytes=self._model_bytes,
#             )
#             proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             if proc.returncode != 0:
#                 raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
#             data = self._parse_last_json(proc.stdout)
#             id_map = {local: active_ids[local] for local in range(len(active_ids))}
#             out = {}
#             for e in data.get('clientResults', []):
#                 local = int(e.get('id', -1))
#                 if local not in id_map:
#                     continue
#                 rx_bytes = float(e.get('rxBytes', 0.0))
#                 done_at  = self._extract_times(e, self._thz_cfg['sim_time'])
#                 thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#                 out[id_map[local]] = {'roundTime': done_at, 'throughput': thr}
#             return out

#         # ---- Wi-Fi/Ethernet via socket controller ----
#         if self.controller == 'socket':
#             res = self._socket_send_round(requestType=requestType, ids=active_ids)
#             out: Dict[int, Dict[str, float]] = {}
#             for e in res:
#                 cid = int(e['id'])
#                 rt = float(e['endTime']) if e['endTime'] is not None else 0.0
#                 thr = float(e['throughput'])
#                 out[cid] = {'roundTime': rt, 'throughput': thr}
#             return out

#         # ---- Wi-Fi/Ethernet standalone (your current flow) ----
#         cmd, exp_Bps = self._wifi_cmd(
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#             use_socket=False,
#         )
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')

#         # Try JSON parse first
#         out: Dict[int, Dict[str, float]] = {}
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         try:
#             data = self._parse_last_json(proc.stdout)
#             for e in data.get('clientResults', []):
#                 local = int(e.get('id', -1))
#                 if local not in id_map:
#                     continue
#                 rx_bytes = float(e.get('rxBytes', 0.0))
#                 rt = self._extract_times(e, 0.0)
#                 if not rt or rt <= 0.0:
#                     rt = (self._model_bytes / max(1.0, exp_Bps))
#                 thr = (rx_bytes / rt) if rt > 0 else 0.0
#                 out[id_map[local]] = {'roundTime': rt, 'throughput': thr}
#             if out:
#                 return out
#         except Exception:
#             pass

#         # Fallback estimate
#         est_rt = (self._model_bytes / max(1.0, exp_Bps))
#         est_thr = exp_Bps
#         for real_id in active_ids:
#             out[real_id] = {'roundTime': est_rt, 'throughput': est_thr}
#         return out

#     # ------------------------------------------------------------------
#     # ASYNC API (for socket, we simulate non-blocking by pre-filling a queue)
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         # bitmap or list → ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         self._async_ids = active_ids
#         self._async_queue = []
#         self._deadline = None

#         if not active_ids:
#             return

#         # THz async: launch and wait like before
#         if self._use_thz:
#             cmd = self._thz_cmd(
#                 total_clients=self.num_clients,
#                 active_count=len(active_ids),
#                 model_bytes=self._model_bytes,
#             )
#             self._proc = subprocess.Popen(
#                 cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )
#             sim_t = self._thz_cfg['sim_time']
#             self._deadline = time.time() + max(10.0, 4.0 * sim_t)
#             return

#         # Socket controller: send once, read all results now, enqueue items
#         if self.controller == 'socket':
#             res = self._socket_send_round(requestType=requestType, ids=active_ids)
#             for e in res:
#                 cid = int(e['id'])
#                 self._async_queue.append({
#                     cid: {
#                         'startTime': float(e['startTime']),
#                         'endTime': float(e['endTime']),
#                         'throughput': float(e['throughput']),
#                     }
#                 })
#             return

#         # Standalone Wi-Fi/Ethernet: spawn process and set deadline
#         cmd, exp_Bps = self._wifi_cmd(active_count=len(active_ids),
#                                       model_bytes=self._model_bytes,
#                                       use_socket=False)
#         self._proc = subprocess.Popen(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         est_rt = self._model_bytes / max(1.0, exp_Bps)
#         self._deadline = time.time() + max(15.0, 4.0 * est_rt)

#     def readAsyncResponse(self):
#         # nothing ever started
#         if self._proc is None and not self._async_queue:
#             return 'end'

#         # Socket controller path: we already prefilled the queue
#         if self.controller == 'socket' and not self._use_thz:
#             if self._async_queue:
#                 return self._async_queue.pop(0)
#             return 'end'

#         # THz / standalone Wi-Fi/Ethernet: keep your previous logic
#         import subprocess

#         if self._proc is not None and self._proc.poll() is None:
#             if self._deadline is not None and time.time() > self._deadline:
#                 try:
#                     self._proc.terminate()
#                 except Exception:
#                     pass
#                 try:
#                     stdout, stderr = self._proc.communicate(timeout=2)
#                 except subprocess.TimeoutExpired:
#                     try:
#                         self._proc.kill()
#                     except Exception:
#                         pass
#                     stdout, stderr = self._proc.communicate()
#                 finally:
#                     self._proc = None

#                 data = {}
#                 try:
#                     data = self._parse_last_json(stdout)
#                 except Exception:
#                     data = {'clientResults': []}

#                 id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#                 results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

#                 # Fallback throughput estimate (Wi-Fi/Ethernet only)
#                 exp_Bps = None
#                 if not self._use_thz:
#                     cfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#                     exp_Bps = _parse_rate_to_Bps(str(cfg.get('data_rate', '250kbps')))

#                 for local in range(len(self._async_ids)):
#                     ent = results.get(local)
#                     real_id = id_map[local]
#                     if ent:
#                         default_t = self._thz_cfg['sim_time'] if self._use_thz else (self._model_bytes / max(1.0, (exp_Bps or 1.0)))
#                         rx_bytes = float(ent.get('rxBytes', 0.0))
#                         done_at  = self._extract_times(ent, default_t)
#                         thr = (rx_bytes / done_at) if done_at and done_at > 0 else (exp_Bps or 0.0)
#                     else:
#                         if self._use_thz:
#                             done_at = self._thz_cfg['sim_time']
#                             thr = 0.0
#                         else:
#                             done_at = (self._model_bytes / max(1.0, (exp_Bps or 1.0)))
#                             thr = (exp_Bps or 0.0)
#                     self._async_queue.append({
#                         real_id: {
#                             'startTime': 0.0,
#                             'endTime': done_at,
#                             'throughput': thr,
#                         }
#                     })
#             else:
#                 return {}

#         if self._proc is not None and self._proc.poll() is not None:
#             stdout, stderr = self._proc.communicate()
#             self._proc = None

#             data = {}
#             parsed = False
#             try:
#                 data = self._parse_last_json(stdout)
#                 parsed = True
#             except Exception:
#                 parsed = False

#             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#             results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])} if parsed else {}

#             exp_Bps = None
#             if not self._use_thz:
#                 cfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#                 exp_Bps = _parse_rate_to_Bps(str(cfg.get('data_rate', '250kbps')))
#                 est_rt = self._model_bytes / max(1.0, exp_Bps)

#             for local in range(len(self._async_ids)):
#                 real_id = id_map[local]
#                 ent = results.get(local)
#                 if ent:
#                     rx_bytes = float(ent.get('rxBytes', 0.0))
#                     done_at  = self._extract_times(ent,
#                                                    self._thz_cfg['sim_time'] if self._use_thz else (self._model_bytes / max(1.0, (exp_Bps or 1.0))))
#                     thr = (rx_bytes / done_at) if done_at and done_at > 0 else (exp_Bps or 0.0)
#                 else:
#                     if self._use_thz:
#                         done_at = self._thz_cfg['sim_time']
#                         thr = 0.0
#                     else:
#                         done_at = est_rt
#                         thr = exp_Bps or 0.0
#                 self._async_queue.append({
#                     real_id: {
#                         'startTime': 0.0,
#                         'endTime': done_at,
#                         'throughput': thr,
#                     }
#                 })

#         if self._async_queue:
#             return self._async_queue.pop(0)
#         return 'end'


# # flsim/network.py — ns-3 THz runner + Wi-Fi/Ethernet (sync + async with timeout & robust parsing)
# import json
# import subprocess
# import time
# from typing import Any, Dict, List, Optional, Tuple

# PATH = '../ns3-fl-network'
# THZ_PROGRAM = 'scratch/thz-macro-central'
# WIFI_PROGRAM = 'wifi_exp'


# def _get(root: Any, path: List[str], default=None):
#     cur = root
#     for k in path:
#         if isinstance(cur, dict) and k in cur:
#             cur = cur[k]
#         elif hasattr(cur, k):
#             cur = getattr(cur, k)
#         else:
#             return default
#     return cur


# def _parse_rate_to_Bps(s: str, default_Bps: float = (250_000 / 8)) -> float:
#     """
#     Parse strings like '250kbps', '6Mbps', '1Gbps' into BYTES/second.
#     SI units (k=1e3). Returns default on failure.
#     """
#     try:
#         s = (s or '').strip().lower()
#         if not s:
#             return default_Bps
#         num = ''
#         unit = ''
#         for ch in s:
#             if ch.isdigit() or ch in '.+-eE':
#                 num += ch
#             else:
#                 unit += ch
#         val = float(num)
#         unit = unit.strip()
#         scale = 1.0
#         if unit in ('bps', ''):
#             scale = 1.0
#         elif unit == 'kbps':
#             scale = 1e3
#         elif unit == 'mbps':
#             scale = 1e6
#         elif unit == 'gbps':
#             scale = 1e9
#         else:
#             # unknown; assume bps
#             scale = 1.0
#         return (val * scale) / 8.0  # bits/s -> bytes/s
#     except Exception:
#         return default_Bps


# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(_get(self.config, ['clients', 'total'], 1))

#         # Which backend? thz (default), wifi, or ethernet
#         self.network_type = str(_get(self.config, ['network', 'type'], 'thz')).lower()
#         self._use_thz = self.network_type not in ('wifi', 'ethernet')

#         # THz sim knobs (safe defaults; override via config.network.thz.*)
#         thz = _get(self.config, ['network', 'thz'], {}) or {}
#         self._thz_cfg = {
#             'carrier_ghz':  float(thz.get('carrier_ghz', 300.0)),  # NEW
#             'pkt_size':     int(thz.get('pkt_size', 600)),
#             'sim_time':     float(thz.get('sim_time', 0.8)),
#             'interval_us':  int(thz.get('interval_us', 20)),
#             'way':          int(thz.get('way', 3)),
#             'radius':       float(thz.get('radius', 0.5)),
#             'beamwidth':    float(thz.get('beamwidth', 40)),
#             'gain':         float(thz.get('gain', 30)),
#             'ap_angle':     float(thz.get('ap_angle', 0)),
#             'sta_angle':    float(thz.get('sta_angle', 180)),
#             'useWhiteList': int(thz.get('useWhiteList', 0)),
#         }

#         # Wi-Fi / Ethernet knobs (read only if used)
#         self._wifi_cfg = _get(self.config, ['network', 'wifi'], {}) or {}
#         self._eth_cfg = _get(self.config, ['network', 'ethernet'], {}) or {}

#         # model bytes per upload
#         # self._model_bytes = int(_get(self.config, ['model', 'size'], 1600))
#         # model.size is in KB in your config -> convert to bytes
#         self._model_bytes = int(_get(self.config, ['model', 'size'], 1600)) * 1024


#         # build ns-3 once
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # async state
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_queue: List[Dict[int, Dict[str, float]]] = []
#         self._deadline: Optional[float] = None  # wall-clock timeout for async job

#     # ------------------------------------------------------------------
#     # compatibility no-ops (old TCP control plane)
#     def connect(self): return
#     def disconnect(self):
#         # nothing persistent (we launch per-request)
#         return

#     # accept list of client objects or raw ids
#     def parse_clients(self, clients):
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     # ------------------------------------------------------------------
#     # core ns-3 launchers
#     def _thz_cmd(self, *, total_clients: int, active_count: int, model_bytes: int) -> List[str]:
#         t = self._thz_cfg
#         return [
#             './ns3', 'run', THZ_PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--CarrierFreqGHz={t["carrier_ghz"]}',  # NEW
#             f'--pktSize={t["pkt_size"]}',
#             f'--simTime={t["sim_time"]}',
#             f'--intervalUs={t["interval_us"]}',
#             f'--way={t["way"]}',
#             f'--radius={t["radius"]}',
#             f'--beamwidth={t["beamwidth"]}',
#             f'--gain={t["gain"]}',
#             f'--apAngle={t["ap_angle"]}',
#             f'--staAngle={t["sta_angle"]}',
#             f'--useWhiteList={t["useWhiteList"]}',
#         ]

#     def _wifi_cmd(self, *, active_count: int, model_bytes: int) -> Tuple[List[str], float]:
#         """
#         Build a cmdline for wifi_exp/ethernet and return (cmd, expected_Bps) where
#         expected_Bps is used for timeout and fallback estimates if no JSON is emitted.
#         """
#         if self.network_type == 'wifi':
#             ncfg = self._wifi_cfg
#         else:
#             ncfg = self._eth_cfg

#         max_pkt   = int(ncfg.get('max_packet_size', 1024))
#         data_rate = str(ncfg.get('data_rate', '250kbps'))
#         tx_gain   = float(ncfg.get('tx_gain', 0.0)) if self.network_type == 'wifi' else None
#         learning  = str(_get(self.config, ['server'], 'sync'))

#         args = [
#             f'--NumClients={active_count}',
#             f'--NetworkType={self.network_type}',
#             f'--ModelSize={model_bytes}',
#             f'--MaxPacketSize={max_pkt}',
#             f'--DataRate={data_rate}',
#             f'--LearningModel={learning}',
#         ]
#         if tx_gain is not None:
#             args.append(f'--TxGain={tx_gain}')

#         cmd = ['./ns3', 'run', WIFI_PROGRAM, '--', *args]
#         exp_Bps = _parse_rate_to_Bps(data_rate)
#         return cmd, exp_Bps

#     @staticmethod
#     def _parse_last_json(stdout: str) -> Dict[str, Any]:
#         last = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     @staticmethod
#     def _extract_times(entry: Dict[str, Any], default_sim_time: float) -> float:
#         # tolerate different field names emitted by the sim
#         if 'doneAt' in entry:
#             return float(entry['doneAt'])
#         if 'endTime' in entry:
#             return float(entry['endTime'])
#         if 'roundTime' in entry:
#             return float(entry['roundTime'])
#         return default_sim_time

#     # ------------------------------------------------------------------
#     # SYNC API
#     def sendRequest(self, *, requestType: int, array: list):
#         # bitmap or list of ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         if not active_ids:
#             return {}

#         if self._use_thz:
#             # ---- THz path (unchanged) ----
#             cmd = self._thz_cmd(
#                 total_clients=self.num_clients,
#                 active_count=len(active_ids),
#                 model_bytes=self._model_bytes,
#             )
#             proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             if proc.returncode != 0:
#                 raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
#             data = self._parse_last_json(proc.stdout)

#             # map local ids 0..N-1 -> real ids
#             id_map = {local: active_ids[local] for local in range(len(active_ids))}
#             out = {}
#             for e in data.get('clientResults', []):
#                 local = int(e.get('id', -1))
#                 if local not in id_map:
#                     continue
#                 rx_bytes = float(e.get('rxBytes', 0.0))
#                 done_at  = self._extract_times(e, self._thz_cfg['sim_time'])
#                 thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#                 out[id_map[local]] = {
#                     'roundTime': done_at,
#                     'throughput': thr,
#                 }
#             return out

#         # ---- Wi-Fi / Ethernet path (binary run; JSON-or-fallback) ----
#         cmd, exp_Bps = self._wifi_cmd(
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#         )
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')

#         # Try JSON parse first
#         out: Dict[int, Dict[str, float]] = {}
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         try:
#             data = self._parse_last_json(proc.stdout)
#             for e in data.get('clientResults', []):
#                 local = int(e.get('id', -1))
#                 if local not in id_map:
#                     continue
#                 rx_bytes = float(e.get('rxBytes', 0.0))
#                 # if wifi_exp emits timings, use them; else estimate from bytes & data rate
#                 rt = self._extract_times(e, 0.0)
#                 if not rt or rt <= 0.0:
#                     rt = (self._model_bytes / max(1.0, exp_Bps))
#                 thr = (rx_bytes / rt) if rt > 0 else 0.0
#                 out[id_map[local]] = {'roundTime': rt, 'throughput': thr}
#             # If JSON existed but had no clientResults, fall back below
#             if out:
#                 return out
#         except Exception:
#             pass

#         # Fallback capacity estimate (keeps training loop moving)
#         est_rt = (self._model_bytes / max(1.0, exp_Bps))
#         est_thr = exp_Bps
#         for i, real_id in enumerate(active_ids):
#             out[real_id] = {'roundTime': est_rt, 'throughput': est_thr}
#         return out

#     # ------------------------------------------------------------------
#     # ASYNC API (with timeout/fallback)
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         # bitmap or list
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         self._async_ids = active_ids
#         self._async_queue = []  # will be filled once process ends
#         self._deadline = None

#         if not active_ids:
#             # nothing to do — synthesize empty and finish
#             self._proc = None
#             self._async_queue = []
#             return

#         if self._use_thz:
#             cmd = self._thz_cmd(
#                 total_clients=self.num_clients,
#                 active_count=len(active_ids),
#                 model_bytes=self._model_bytes,
#             )
#             self._proc = subprocess.Popen(
#                 cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )
#             # Allow plenty of margin over sim_time; never block forever
#             sim_t = self._thz_cfg['sim_time']
#             self._deadline = time.time() + max(10.0, 4.0 * sim_t)
#             return

#         # Wi-Fi / Ethernet: run wifi_exp; deadline from data rate (estimated transfer time * margin)
#         cmd, exp_Bps = self._wifi_cmd(active_count=len(active_ids), model_bytes=self._model_bytes)
#         self._proc = subprocess.Popen(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         est_rt = self._model_bytes / max(1.0, exp_Bps)
#         self._deadline = time.time() + max(15.0, 4.0 * est_rt)

#     def readAsyncResponse(self):
#         """
#         Poll once:
#         - returns {} while ns-3 is running
#         - then returns one client’s dict at a time: {id: {...}}
#         - when all delivered, returns 'end'
#         - if deadline exceeded, kill process and synthesize results
#         """
#         import subprocess

#         # nothing ever started
#         if self._proc is None and not self._async_queue:
#             return 'end'

#         # still running
#         if self._proc is not None and self._proc.poll() is None:
#             # timeout guard
#             if self._deadline is not None and time.time() > self._deadline:
#                 try:
#                     self._proc.terminate()
#                 except Exception:
#                     pass
#                 # try graceful, then hard kill
#                 try:
#                     stdout, stderr = self._proc.communicate(timeout=2)
#                 except subprocess.TimeoutExpired:
#                     try:
#                         self._proc.kill()
#                     except Exception:
#                         pass
#                     stdout, stderr = self._proc.communicate()
#                 finally:
#                     self._proc = None

#                 # Try to parse output; if none, synthesize "sim finished" entries
#                 data = {}
#                 try:
#                     data = self._parse_last_json(stdout)
#                 except Exception:
#                     data = {'clientResults': []}

#                 id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#                 results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

#                 # Estimate fallback throughput if needed (Wi-Fi/Ethernet)
#                 exp_Bps = None
#                 if not self._use_thz:
#                     # reconstruct expected Bps roughly from config
#                     if self.network_type == 'wifi':
#                         cfg = self._wifi_cfg
#                     else:
#                         cfg = self._eth_cfg
#                     exp_Bps = _parse_rate_to_Bps(str(cfg.get('data_rate', '250kbps')))

#                 for local in range(len(self._async_ids)):
#                     ent = results.get(local)
#                     real_id = id_map[local]
#                     if ent:
#                         if self._use_thz:
#                             default_t = self._thz_cfg['sim_time']
#                         else:
#                             default_t = (self._model_bytes / max(1.0, (exp_Bps or 1.0)))
#                         rx_bytes = float(ent.get('rxBytes', 0.0))
#                         done_at  = self._extract_times(ent, default_t)
#                         thr = (rx_bytes / done_at) if done_at and done_at > 0 else (exp_Bps or 0.0)
#                     else:
#                         if self._use_thz:
#                             done_at = self._thz_cfg['sim_time']
#                             thr = 0.0
#                         else:
#                             done_at = (self._model_bytes / max(1.0, (exp_Bps or 1.0)))
#                             thr = (exp_Bps or 0.0)
#                     self._async_queue.append({
#                         real_id: {
#                             'startTime': 0.0,
#                             'endTime': done_at,
#                             'throughput': thr,
#                         }
#                     })
#                 # fall through to serve queue
#             else:
#                 return {}

#         # finished: if queue not built yet, parse and build single-client chunks
#         if self._proc is not None and self._proc.poll() is not None:
#             stdout, stderr = self._proc.communicate()
#             self._proc = None

#             data = {}
#             parsed = False
#             try:
#                 data = self._parse_last_json(stdout)
#                 parsed = True
#             except Exception:
#                 parsed = False

#             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#             results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])} if parsed else {}

#             # estimate Bps if needed (Wi-Fi/Ethernet)
#             exp_Bps = None
#             if not self._use_thz:
#                 cfg = self._wifi_cfg if self.network_type == 'wifi' else self._eth_cfg
#                 exp_Bps = _parse_rate_to_Bps(str(cfg.get('data_rate', '250kbps')))
#                 est_rt = self._model_bytes / max(1.0, exp_Bps)

#             for local in range(len(self._async_ids)):
#                 real_id = id_map[local]
#                 ent = results.get(local)
#                 if ent:
#                     rx_bytes = float(ent.get('rxBytes', 0.0))
#                     done_at  = self._extract_times(ent,
#                                                    self._thz_cfg['sim_time'] if self._use_thz else (self._model_bytes / max(1.0, (exp_Bps or 1.0))))
#                     thr = (rx_bytes / done_at) if done_at and done_at > 0 else (exp_Bps or 0.0)
#                 else:
#                     if self._use_thz:
#                         done_at = self._thz_cfg['sim_time']
#                         thr = 0.0
#                     else:
#                         done_at = est_rt
#                         thr = exp_Bps or 0.0
#                 self._async_queue.append({
#                     real_id: {
#                         'startTime': 0.0,
#                         'endTime': done_at,
#                         'throughput': thr,
#                     }
#                 })

#         # serve one and pop
#         if self._async_queue:
#             return self._async_queue.pop(0)

#         return 'end'


# # flsim/network.py — ns-3 THz runner (sync + async with timeout & robust parsing)
# import json
# import subprocess
# import time
# from typing import Any, Dict, List, Optional

# PATH = '../ns3-fl-network'
# PROGRAM = 'scratch/thz-macro-central'


# def _get(root: Any, path: List[str], default=None):
#     cur = root
#     for k in path:
#         if isinstance(cur, dict) and k in cur:
#             cur = cur[k]
#         elif hasattr(cur, k):
#             cur = getattr(cur, k)
#         else:
#             return default
#     return cur


# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(_get(self.config, ['clients', 'total'], 1))

#         # THz sim knobs (safe defaults; override via config.network.thz.*)
#         thz = _get(self.config, ['network', 'thz'], {}) or {}
#         self._thz_cfg = {
#             'carrier_ghz':  float(thz.get('carrier_ghz', 300.0)),  # NEW
#             'pkt_size':     int(thz.get('pkt_size', 600)),
#             'sim_time':     float(thz.get('sim_time', 0.8)),
#             'interval_us':  int(thz.get('interval_us', 20)),
#             'way':          int(thz.get('way', 3)),
#             'radius':       float(thz.get('radius', 0.5)),
#             'beamwidth':    float(thz.get('beamwidth', 40)),
#             'gain':         float(thz.get('gain', 30)),
#             'ap_angle':     float(thz.get('ap_angle', 0)),
#             'sta_angle':    float(thz.get('sta_angle', 180)),
#             'useWhiteList': int(thz.get('useWhiteList', 0)),
#         }

#         # model bytes per upload
#         self._model_bytes = int(_get(self.config, ['model', 'size'], 1600))

#         # build ns-3 once
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # async state
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_queue: List[Dict[int, Dict[str, float]]] = []
#         self._deadline: Optional[float] = None  # wall-clock timeout for async job

#     # ------------------------------------------------------------------
#     # compatibility no-ops (old TCP control plane)
#     def connect(self): return
#     def disconnect(self): return

#     # accept list of client objects or raw ids
#     def parse_clients(self, clients):
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     # ------------------------------------------------------------------
#     # core ns-3 launchers
#     def _cmd(self, *, total_clients: int, active_count: int, model_bytes: int) -> List[str]:
#         t = self._thz_cfg
#         return [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--CarrierFreqGHz={t["carrier_ghz"]}',  # NEW
#             f'--pktSize={t["pkt_size"]}',
#             f'--simTime={t["sim_time"]}',
#             f'--intervalUs={t["interval_us"]}',
#             f'--way={t["way"]}',
#             f'--radius={t["radius"]}',
#             f'--beamwidth={t["beamwidth"]}',
#             f'--gain={t["gain"]}',
#             f'--apAngle={t["ap_angle"]}',
#             f'--staAngle={t["sta_angle"]}',
#             f'--useWhiteList={t["useWhiteList"]}',
#         ]

#     @staticmethod
#     def _parse_last_json(stdout: str) -> Dict[str, Any]:
#         last = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     @staticmethod
#     def _extract_times(entry: Dict[str, Any], default_sim_time: float) -> float:
#         # tolerate different field names emitted by the sim
#         if 'doneAt' in entry:
#             return float(entry['doneAt'])
#         if 'endTime' in entry:
#             return float(entry['endTime'])
#         if 'roundTime' in entry:
#             return float(entry['roundTime'])
#         return default_sim_time

#     # ------------------------------------------------------------------
#     # SYNC API
#     def sendRequest(self, *, requestType: int, array: list):
#         # bitmap or list of ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         if not active_ids:
#             return {}

#         cmd = self._cmd(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#         )
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
#         data = self._parse_last_json(proc.stdout)

#         # map local ids 0..N-1 -> real ids
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         out = {}
#         for e in data.get('clientResults', []):
#             local = int(e.get('id', -1))
#             if local not in id_map:
#                 continue
#             rx_bytes = float(e.get('rxBytes', 0.0))
#             done_at  = self._extract_times(e, self._thz_cfg['sim_time'])
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             out[id_map[local]] = {
#                 'roundTime': done_at,
#                 'throughput': thr,
#             }
#         return out

#     # ------------------------------------------------------------------
#     # ASYNC API (with timeout/fallback)
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         # bitmap or list
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         self._async_ids = active_ids
#         self._async_queue = []  # will be filled once process ends
#         self._deadline = None

#         if not active_ids:
#             # nothing to do — synthesize empty and finish
#             self._proc = None
#             self._async_queue = []
#             return

#         cmd = self._cmd(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#         )
#         self._proc = subprocess.Popen(
#             cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         # Allow plenty of margin over sim_time; never block forever
#         sim_t = self._thz_cfg['sim_time']
#         self._deadline = time.time() + max(10.0, 4.0 * sim_t)

#     def readAsyncResponse(self):
#         """
#         Poll once:
#         - returns {} while ns-3 is running
#         - then returns one client’s dict at a time: {id: {...}}
#         - when all delivered, returns 'end'
#         - if deadline exceeded, kill process and synthesize results
#         """
#         import subprocess

#         # nothing ever started
#         if self._proc is None and not self._async_queue:
#             return 'end'

#         # still running
#         if self._proc is not None and self._proc.poll() is None:
#             # timeout guard
#             if self._deadline is not None and time.time() > self._deadline:
#                 try:
#                     self._proc.terminate()
#                 except Exception:
#                     pass
#                 # try graceful, then hard kill
#                 try:
#                     stdout, stderr = self._proc.communicate(timeout=2)
#                 except subprocess.TimeoutExpired:
#                     try:
#                         self._proc.kill()
#                     except Exception:
#                         pass
#                     stdout, stderr = self._proc.communicate()
#                 finally:
#                     self._proc = None

#                 # Try to parse output; if none, synthesize "sim finished" entries
#                 data = {}
#                 try:
#                     data = self._parse_last_json(stdout)
#                 except Exception:
#                     data = {'clientResults': []}

#                 id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#                 results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

#                 # build per-client responses; synthesize missing with sim_time
#                 for local in range(len(self._async_ids)):
#                     ent = results.get(local)
#                     real_id = id_map[local]
#                     if ent:
#                         rx_bytes = float(ent.get('rxBytes', 0.0))
#                         done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
#                         thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#                     else:
#                         done_at = self._thz_cfg['sim_time']
#                         thr = 0.0
#                     self._async_queue.append({
#                         real_id: {
#                             'startTime': 0.0,
#                             'endTime': done_at,
#                             'throughput': thr,
#                         }
#                     })
#                 # fall through to serve queue
#             else:
#                 return {}

#         # finished: if queue not built yet, parse and build single-client chunks
#         if self._proc is not None and self._proc.poll() is not None:
#             stdout, stderr = self._proc.communicate()
#             self._proc = None

#             data = self._parse_last_json(stdout)
#             # map local -> real ids
#             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#             results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

#             # build per-client responses in a queue
#             for local in range(len(self._async_ids)):
#                 ent = results.get(local)
#                 if not ent:
#                     continue
#                 real_id = id_map[local]
#                 rx_bytes = float(ent.get('rxBytes', 0.0))
#                 done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
#                 thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#                 self._async_queue.append({
#                     real_id: {
#                         'startTime': 0.0,
#                         'endTime': done_at,
#                         'throughput': thr,
#                     }
#                 })

#         # serve one and pop
#         if self._async_queue:
#             return self._async_queue.pop(0)

#         return 'end'


    # def readAsyncResponse(self):
    #     """
    #     Poll once:
    #       - returns {} while ns-3 is running
    #       - then returns one client’s dict at a time: {id: {...}}
    #       - when all delivered, returns 'end'
    #       - if deadline exceeded, kill process and synthesize results
    #     """
    #     # nothing ever started
    #     if self._proc is None and not self._async_queue:
    #         return 'end'

    #     # still running
    #     if self._proc is not None and self._proc.poll() is None:
    #         # timeout guard
    #         if self._deadline is not None and time.time() > self._deadline:
    #             try:
    #                 self._proc.terminate()
    #             except Exception:
    #                 pass
    #             stdout = self._proc.stdout.read() if self._proc.stdout else ''
    #             stderr = self._proc.stderr.read() if self._proc.stderr else ''
    #             self._proc = None

    #             # Try to parse output; if none, synthesize "sim finished" entries
    #             data = {}
    #             try:
    #                 data = self._parse_last_json(stdout)
    #             except Exception:
    #                 data = {'clientResults': []}

    #             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
    #             results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

    #             # build per-client responses; synthesize missing with sim_time
    #             for local in range(len(self._async_ids)):
    #                 ent = results.get(local)
    #                 real_id = id_map[local]
    #                 if ent:
    #                     rx_bytes = float(ent.get('rxBytes', 0.0))
    #                     done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
    #                     thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
    #                 else:
    #                     done_at = self._thz_cfg['sim_time']
    #                     thr = 0.0
    #                 self._async_queue.append({
    #                     real_id: {
    #                         'startTime': 0.0,
    #                         'endTime': done_at,
    #                         'throughput': thr,
    #                     }
    #                 })
    #             # fall through to serve queue
    #         else:
    #             return {}

    #     # finished: if queue not built yet, parse and build single-client chunks
    #     if self._proc is not None:
    #         stdout = self._proc.stdout.read() if self._proc.stdout else ''
    #         stderr = self._proc.stderr.read() if self._proc.stderr else ''
    #         self._proc = None

    #         data = self._parse_last_json(stdout)
    #         # map local -> real ids
    #         id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
    #         results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

    #         # build per-client responses in a queue
    #         for local in range(len(self._async_ids)):
    #             ent = results.get(local)
    #             if not ent:
    #                 continue
    #             real_id = id_map[local]
    #             rx_bytes = float(ent.get('rxBytes', 0.0))
    #             done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
    #             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
    #             self._async_queue.append({
    #                 real_id: {
    #                     'startTime': 0.0,
    #                     'endTime': done_at,
    #                     'throughput': thr,
    #                 }
    #             })

    #     # serve one and pop
    #     if self._async_queue:
    #         return self._async_queue.pop(0)

    #     return 'end'


# # flsim/network.py — ns-3 THz runner (sync + async)
# import json
# import subprocess
# from typing import Any, Dict, List, Optional

# PATH = '../ns3-fl-network'
# PROGRAM = 'scratch/thz-macro-central'


# def _get(root: Any, path: List[str], default=None):
#     cur = root
#     for k in path:
#         if isinstance(cur, dict) and k in cur:
#             cur = cur[k]
#         elif hasattr(cur, k):
#             cur = getattr(cur, k)
#         else:
#             return default
#     return cur


# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(_get(self.config, ['clients', 'total'], 1))

#         # THz sim knobs (safe defaults; override via config.network.thz.*)
#         thz = _get(self.config, ['network', 'thz'], {}) or {}
#         self._thz_cfg = {
#             'pkt_size':    int(thz.get('pkt_size', 600)),
#             'sim_time':    float(thz.get('sim_time', 0.8)),
#             'interval_us': int(thz.get('interval_us', 20)),
#             'way':         int(thz.get('way', 3)),
#             'radius':      float(thz.get('radius', 0.5)),
#             'beamwidth':   float(thz.get('beamwidth', 40)),
#             'gain':        float(thz.get('gain', 30)),
#             'ap_angle':    float(thz.get('ap_angle', 0)),
#             'sta_angle':   float(thz.get('sta_angle', 180)),
#             'useWhiteList':int(thz.get('useWhiteList', 0)),
#         }

#         # model bytes per upload
#         self._model_bytes = int(_get(self.config, ['model', 'size'], 1600))

#         # build ns-3 once
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # async state
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_queue: List[Dict[int, Dict[str, float]]] = []

#     # ------------------------------------------------------------------
#     # compatibility no-ops (old TCP control plane)
#     def connect(self): return
#     def disconnect(self): return

#     # accept list of client objects or raw ids
#     def parse_clients(self, clients):
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     # ------------------------------------------------------------------
#     # core ns-3 launchers
#     def _cmd(self, *, total_clients: int, active_count: int, model_bytes: int) -> List[str]:
#         t = self._thz_cfg
#         return [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={t["pkt_size"]}',
#             f'--simTime={t["sim_time"]}',
#             f'--intervalUs={t["interval_us"]}',
#             f'--way={t["way"]}',
#             f'--radius={t["radius"]}',
#             f'--beamwidth={t["beamwidth"]}',
#             f'--gain={t["gain"]}',
#             f'--apAngle={t["ap_angle"]}',
#             f'--staAngle={t["sta_angle"]}',
#             f'--useWhiteList={t["useWhiteList"]}',
#         ]

#     @staticmethod
#     def _parse_last_json(stdout: str) -> Dict[str, Any]:
#         last = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     # ------------------------------------------------------------------
#     # SYNC API
#     def sendRequest(self, *, requestType: int, array: list):
#         # bitmap or list of ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         if not active_ids:
#             return {}

#         cmd = self._cmd(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#         )
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
#         data = self._parse_last_json(proc.stdout)

#         # map local ids 0..N-1 -> real ids
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         out = {}
#         for e in data.get('clientResults', []):
#             local = int(e.get('id', -1))
#             if local not in id_map:
#                 continue
#             rx_bytes = float(e.get('rxBytes', 0.0))
#             done_at  = float(e.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             out[id_map[local]] = {
#                 'roundTime': done_at if done_at >= 0 else self._thz_cfg['sim_time'],
#                 'throughput': thr,
#             }
#         return out

#     # ------------------------------------------------------------------
#     # ASYNC API
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         # bitmap or list
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         self._async_ids = active_ids
#         self._async_queue = []  # will be filled once process ends

#         if not active_ids:
#             # nothing to do — synthesize empty and finish
#             self._proc = None
#             self._async_queue = []
#             return

#         cmd = self._cmd(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#         )
#         self._proc = subprocess.Popen(
#             cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )

#     def readAsyncResponse(self):
#         """
#         Poll once:
#           - returns {} while ns-3 is running
#           - then returns one client’s dict at a time: {id: {...}}
#           - when all delivered, returns 'end'
#         """
#         # nothing ever started
#         if self._proc is None and not self._async_queue:
#             return 'end'

#         # still running
#         if self._proc is not None and self._proc.poll() is None:
#             return {}

#         # finished: if queue not built yet, parse and build single-client chunks
#         if self._proc is not None:
#             stdout = self._proc.stdout.read() if self._proc.stdout else ''
#             stderr = self._proc.stderr.read() if self._proc.stderr else ''
#             self._proc = None

#             data = self._parse_last_json(stdout)
#             # map local -> real ids
#             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#             results = {int(e['id']): e for e in data.get('clientResults', [])}

#             # build per-client responses in a queue
#             for local in range(len(self._async_ids)):
#                 ent = results.get(local)
#                 if not ent:
#                     continue
#                 real_id = id_map[local]
#                 rx_bytes = float(ent.get('rxBytes', 0.0))
#                 done_at  = float(ent.get('doneAt', -1.0))
#                 thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#                 self._async_queue.append({
#                     real_id: {
#                         'startTime': 0.0,
#                         'endTime': done_at if done_at >= 0 else self._thz_cfg['sim_time'],
#                         'throughput': thr,
#                     }
#                 })

#         # serve one and pop
#         if self._async_queue:
#             return self._async_queue.pop(0)

#         return 'end'


# # network.py — ns-3 THz runner (drop-in)
# import json
# import subprocess

# PATH = '../ns3-fl-network'
# PROGRAM = 'scratch/thz-macro-central'

# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(self.config.clients.total)

#         # Build ns-3 once up-front
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#     def connect(self):    # no TCP control plane
#         return

#     def disconnect(self):
#         return

#     def parse_clients(self, clients):
#         # Accept list of client objects or raw ids
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     def _get_thz_cfg(self):
#         # Safe defaults + allow overrides from config.json
#         thz = getattr(self.config.network, 'thz', {})
#         g = lambda k, d: thz.get(k, d) if isinstance(thz, dict) else getattr(thz, k, d)
#         return {
#             'pkt_size':   int(g('pkt_size', 600)),
#             'sim_time':   float(g('sim_time', 0.8)),
#             'interval_us':int(g('interval_us', 20)),
#             'way':        int(g('way', 3)),
#             'radius':     float(g('radius', 0.5)),
#             'beamwidth':  float(g('beamwidth', 40)),
#             'gain':       float(g('gain', 30)),
#             'ap_angle':   float(g('ap_angle', 0)),
#             'sta_angle':  float(g('sta_angle', 180)),
#             'useWhiteList': int(g('useWhiteList', 0)),
#         }

#     def _run_ns3(self, *, total_clients: int, active_count: int, model_bytes: int):
#         thz = self._get_thz_cfg()
#         cmd = [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={thz["pkt_size"]}',
#             f'--simTime={thz["sim_time"]}',
#             f'--intervalUs={thz["interval_us"]}',
#             f'--way={thz["way"]}',
#             f'--radius={thz["radius"]}',
#             f'--beamwidth={thz["beamwidth"]}',
#             f'--gain={thz["gain"]}',
#             f'--apAngle={thz["ap_angle"]}',
#             f'--staAngle={thz["sta_angle"]}',
#             f'--useWhiteList={thz["useWhiteList"]}',
#         ]
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')

#         # Find the last JSON line
#         last = None
#         for line in reversed(proc.stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     def sendRequest(self, *, requestType: int, array: list):
#         # Accept either bitmap or list of ids
#         if len(array) == self.num_clients and all(x in (0,1) for x in array):
#             active_ids = [i for i, f in enumerate(array) if f]
#         else:
#             active_ids = self.parse_clients(array)
#         if not active_ids:
#             return {}

#         # Model bytes from config (bytes per client upload)
#         model_bytes = int(self.config.model.size)
#         data = self._run_ns3(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=model_bytes,
#         )

#         # ns-3 returns entries indexed 0..active_count-1; map back to real ids
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         ret = {}
#         for e in data.get('clientResults', []):
#             local_id = int(e['id'])
#             real_id = id_map.get(local_id, local_id)
#             rx_bytes = float(e.get('rxBytes', 0.0))
#             done_at  = float(e.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             ret[real_id] = {'roundTime': done_at if done_at >= 0 else self._get_thz_cfg()['sim_time'],
#                             'throughput': thr}
#         return ret

#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         # Not wired for async yet
#         raise NotImplementedError('Async path not implemented for THz runner.')

#     def readAsyncResponse(self):
#         raise NotImplementedError('Async path not implemented for THz runner.')


# # flsim/network.py
# import json
# import subprocess

# PATH = '../ns3-fl-network'
# PROGRAM = 'scratch/thz-macro-central'

# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = self.config.clients.total

#         # ns-3 knobs (from config, with safe defaults)
#         self.pkt_size = getattr(self.config.network.thz, 'pkt_size', 600)
#         self.sim_time = getattr(self.config.network.thz, 'sim_time', 0.5)
#         self.handshake_way = getattr(self.config.network.thz, 'way', 3)
#         self.interval_us = getattr(self.config.network.thz, 'interval_us', 20)
#         self.use_white_list = int(getattr(self.config.network.thz, 'use_white_list', 0))
#         self.ap_angle = getattr(self.config.network.thz, 'ap_angle', 0)
#         self.sta_angle = getattr(self.config.network.thz, 'sta_angle', 180)
#         self.radius = getattr(self.config.network.thz, 'radius', 0.5)
#         self.beamwidth = getattr(self.config.network.thz, 'beamwidth', 40)
#         self.gain = getattr(self.config.network.thz, 'gain', 30)

#         # Build once
#         proc = subprocess.run(
#             './ns3 build', cwd=PATH, shell=True, text=True,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}\n{proc.stdout}')

#     # API compatibility
#     def connect(self): return
#     def disconnect(self): return

#     def parse_clients(self, clients):
#         # return actual selected IDs
#         return [c.client_id for c in clients]

#     def _run_ns3(self, *, total_clients: int, active_ids: list, model_bytes: int):
#         """
#         Launch ns-3 and return parsed JSON (the program prints {"clientResults":[...]}).
#         We map local ids [0..N-1] to your selected client IDs.
#         """
#         cmd = [
#             './ns3','run',PROGRAM,'--',
#             f'--nodeNum={total_clients}',
#             f'--clients={len(active_ids)}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={self.pkt_size}',
#             f'--simTime={self.sim_time}',
#             f'--way={self.handshake_way}',
#             f'--intervalUs={self.interval_us}',
#             f'--useWhiteList={self.use_white_list}',
#             f'--apAngle={self.ap_angle}',
#             f'--staAngle={self.sta_angle}',
#             f'--radius={self.radius}',
#             f'--beamwidth={self.beamwidth}',
#             f'--gain={self.gain}',
#         ]
#         proc = subprocess.run(cmd, cwd=PATH, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\n{proc.stderr}\n{proc.stdout}')

#         # parse the last JSON line
#         last_json = None
#         for line in reversed(proc.stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last_json = s
#                 break
#         if not last_json:
#             raise RuntimeError(f'No JSON in ns-3 output.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
#         return json.loads(last_json)

#     def sendRequest(self, *, requestType: int, array: list):
#         # accept ID list or bitmap
#         if len(array) == self.num_clients and all(x in (0,1) for x in array):
#             active_ids = [i for i,v in enumerate(array) if v]
#         else:
#             active_ids = list(map(int, array))
#         if not active_ids:
#             return {}

#         data = self._run_ns3(
#             total_clients=self.num_clients,
#             active_ids=active_ids,
#             model_bytes=int(self.config.model.size)
#         )

#         # ns-3 emits local ids 0..N-1; map back to your chosen client IDs
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         ret = {}
#         for ent in data.get('clientResults', []):
#             local_id = int(ent.get('id', -1))
#             if local_id not in id_map:
#                 continue
#             rx_bytes = float(ent.get('rxBytes', 0.0))
#             done_at  = float(ent.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             ret[id_map[local_id]] = {
#                 'roundTime': done_at if done_at >= 0 else self.sim_time,
#                 'throughput': thr
#             }
#         return ret

#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         raise NotImplementedError('Async path not wired for THz runner.')

#     def readAsyncResponse(self):
#         raise NotImplementedError('Async path not wired for THz runner.')



# # network.py — THz (ns-3) adapter for flsim
# import json
# import subprocess
# import shlex
# from typing import List, Dict, Any, Optional

# # Path to your ns-3 tree (the one with the ./ns3 helper script)
# PATH = '../ns3-fl-network'
# # Your ns-3 program
# PROGRAM = 'scratch/thz-macro-central'


# def _get_nested(root: Any, path: List[str], default=None):
#     """Works with both object-style and dict-style config objects."""
#     cur = root
#     for key in path:
#         if hasattr(cur, key):
#             cur = getattr(cur, key)
#         elif isinstance(cur, dict) and key in cur:
#             cur = cur[key]
#         else:
#             return default
#     return cur


# class Network(object):
#     def __init__(self, config):
#         self.config = config

#         # ---- read config (keeps MNIST + everything else untouched) ----
#         self.num_clients = int(_get_nested(self.config, ['clients', 'total'], 1))
#         self.network_type = _get_nested(self.config, ['network', 'type'], 'thz')

#         if self.network_type != 'thz':
#             raise ValueError('Set network.type to "thz" in config.json to use the THz ns-3 runner.')

#         thz_cfg = _get_nested(self.config, ['network', 'thz'], {}) or {}
#         # defaults safe for your current runs; override in config.network.thz
#         self.pkt_size = int(thz_cfg.get('pkt_size', 600))
#         self.sim_time = float(thz_cfg.get('sim_time', 0.002))
#         self.handshake_way = int(thz_cfg.get('way', 0))

#         # Model upload size (bytes) for this FL round
#         self.model_bytes = int(_get_nested(self.config, ['model', 'size'], 1600))

#         # ---- build ns-3 once ----
#         proc = subprocess.Popen(
#             './ns3 build',
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True,
#             cwd=PATH,
#         )
#         proc.wait()
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # ---- async state (if you use "server": "async") ----
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_result: Optional[Dict[int, Dict[str, float]]] = None
#         self._async_served_once: bool = False

#     # Keep API used by flsim; we don’t need a control socket anymore.
#     def connect(self):
#         return

#     def disconnect(self):
#         return

#     def parse_clients(self, clients):
#         """Return a 0/1 bitmap (original behaviour) so upstream code stays happy."""
#         bitmap = [0 for _ in range(self.num_clients)]
#         for c in clients:
#             bitmap[c.client_id] = 1
#         return bitmap

#     # ---------- core runner ----------
#     def _run_ns3_once(self, total_clients: int, active_count: int, model_bytes: int) -> Dict[str, Any]:
#         cmd = [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={self.pkt_size}',
#             f'--simTime={self.sim_time}',
#             f'--way={self.handshake_way}',
#         ]
#         proc = subprocess.run(
#             cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(
#                 'ns-3 run failed:\n'
#                 f'CMD: {shlex.join(cmd)}\n'
#                 f'STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}'
#             )

#         # Find the last JSON-looking line
#         last_json = None
#         for line in reversed(proc.stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last_json = s
#                 break
#         if not last_json:
#             raise RuntimeError(
#                 'No JSON summary found in ns-3 output.\n'
#                 f'CMD: {shlex.join(cmd)}\n'
#                 f'STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}'
#             )
#         return json.loads(last_json)

#     # ---------- sync path ----------
#     def sendRequest(self, *, requestType: int, array: list):
#         """
#         Returns { client_id: {"roundTime": <sec>, "throughput": <bytes/sec>} }.
#         Accepts either a bitmap (len = total clients) or a list of IDs in `array`.
#         """
#         # Accept bitmap or list-of-ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = list(map(int, array))

#         if not active_ids:
#             return {}

#         per_round = len(active_ids)
#         data = self._run_ns3_once(
#             total_clients=self.num_clients,
#             active_count=per_round,
#             model_bytes=self.model_bytes,
#         )

#         # ns-3 emits ids 0..per_round-1; map back to selected client IDs
#         id_map = {local_i: active_ids[local_i] for local_i in range(per_round)}

#         results_by_local = {int(e['id']): e for e in data.get('clientResults', [])}
#         out = {}
#         for local_i in range(per_round):
#             ent = results_by_local.get(local_i)
#             if not ent:
#                 continue
#             rx_bytes = float(ent.get('rxBytes', 0.0))
#             done_at = float(ent.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             out[id_map[local_i]] = {
#                 'roundTime': done_at if done_at >= 0 else self.sim_time,
#                 'throughput': thr,
#             }
#         return out

#     # ---------- async path (simple adapter) ----------
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         """
#         Fire-and-return: launch ns-3 in the background for this set of clients.
#         On completion, readAsyncResponse() will return a dict once, then 'end'.
#         """
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         # Accept bitmap or list-of-ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = list(map(int, array))
#         if not active_ids:
#             # Nothing to do; synthesize immediate "end"
#             self._async_ids = []
#             self._async_result = {}
#             self._proc = None
#             self._async_served_once = True
#             return

#         per_round = len(active_ids)
#         self._async_ids = active_ids
#         self._async_served_once = False
#         self._async_result = None

#         cmd = [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={self.num_clients}',
#             f'--clients={per_round}',
#             f'--modelBytes={self.model_bytes}',
#             f'--pktSize={self.pkt_size}',
#             f'--simTime={self.sim_time}',
#             f'--way={self.handshake_way}',
#         ]
#         self._proc = subprocess.Popen(
#             cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )

#     def readAsyncResponse(self):
#         """
#         Poll once. While ns-3 is still running -> return {}.
#         When it finishes -> return a dict {id: {startTime, endTime, throughput}} once,
#         and on the next call return 'end'.
#         """
#         if self._proc is None:
#             # Either nothing was started or we've already handed out the result + 'end'
#             return 'end' if self._async_served_once else {}

#         retcode = self._proc.poll()
#         if retcode is None:
#             # Still running
#             return {}

#         # Finished: parse once
#         stdout = self._proc.stdout.read() if self._proc.stdout else ''
#         stderr = self._proc.stderr.read() if self._proc.stderr else ''
#         self._proc = None

#         # Find last JSON line
#         last_json = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last_json = s
#                 break
#         if not last_json:
#             raise RuntimeError(f'Async ns-3 produced no JSON.\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}')

#         data = json.loads(last_json)
#         per_round = len(self._async_ids)
#         id_map = {local_i: self._async_ids[local_i] for local_i in range(per_round)}
#         results_by_local = {int(e['id']): e for e in data.get('clientResults', [])}

#         # fabricate startTime=0.0 and endTime=doneAt (works with async server expectations)
#         out = {}
#         for local_i in range(per_round):
#             ent = results_by_local.get(local_i)
#             if not ent:
#                 continue
#             rx_bytes = float(ent.get('rxBytes', 0.0))
#             done_at = float(ent.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             out[id_map[local_i]] = {
#                 'startTime': 0.0,
#                 'endTime': done_at if done_at >= 0 else self.sim_time,
#                 'throughput': thr,
#             }

#         self._async_result = out
#         self._async_served_once = False  # not yet served to caller
#         # Hand it out now; next poll will return 'end'
#         self._async_served_once = True
#         return out

# Original code commented out for reference; not used in the new implementation.


# #from py_interface import *
# from ctypes import *
# import socket
# import struct
# import subprocess
# import json
# import re

# TCP_IP = '127.0.0.1'
# TCP_PORT = 8080
# PATH='../ns3-fl-network'
# # PROGRAM='wifi_exp'
# PROGRAM='scratch/thz-macro-central'

# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = self.config.clients.total
#         self.network_type = self.config.network.type

#         proc = subprocess.Popen('./ns3 build', shell=True, stdout=subprocess.PIPE,
#                                 universal_newlines=True, cwd=PATH)
#         proc.wait()
#         if proc.returncode != 0:
#             exit(-1)

#         command = './ns3 run "' + PROGRAM + ' --NumClients=' + str(self.num_clients) + ' --NetworkType=' + self.network_type
#         command += ' --ModelSize=' + str(self.config.model.size)
#         '''print(self.config.network)
#         for net in self.config.network:
#             if net == self.network_type:
#                 print(net.items())'''

#         if self.network_type == 'wifi':
#             command += ' --TxGain=' + str(self.config.network.wifi['tx_gain'])
#             command += ' --MaxPacketSize=' + str(self.config.network.wifi['max_packet_size'])
#         else: # else assume ethernet
#             command += ' --MaxPacketSize=' + str(self.config.network.ethernet['max_packet_size'])

#         command += " --LearningModel=" + str(self.config.server)

#         command += '"'
#         print(command)

#         proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
#                                 universal_newlines=True, cwd=PATH)


#     def parse_clients(self, clients):
#         clients_to_send = [0 for _ in range(self.num_clients)]
#         for client in clients:
#             clients_to_send[client.client_id] = 1
#         return clients_to_send

#     def connect(self):
#         self.s = socket.create_connection((TCP_IP, TCP_PORT,))

#     def sendRequest(self, *, requestType: int, array: list):
#         print("sending")
#         print(array)
#         message = struct.pack("II", requestType, len(array))
#         self.s.send(message)
#         # for the total number of clients
#         # is the index in lit at client.id equal
#         for ele in array:
#             self.s.send(struct.pack("I", ele))

#         resp = self.s.recv(8)
#         print("resp")
#         print(resp)
#         if len(resp) < 8:
#             print(len(resp), resp)
#         command, nItems = struct.unpack("II", resp)
#         ret = {}
#         for i in range(nItems):
#             dr = self.s.recv(8 * 3)
#             eid, roundTime, throughput = struct.unpack("Qdd", dr)
#             temp = {"roundTime": roundTime, "throughput": throughput}
#             ret[eid] = temp
#         return ret

#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         print("sending")
#         print(array)
#         message = struct.pack("II",requestType , len(array))
#         self.s.send(message)
#         # for the total number of clients
#         # is the index in lit at client.id equal
#         for ele in array:
#             self.s.send(struct.pack("I", ele))

#     def readAsyncResponse(self):
#         resp = self.s.recv(8)
#         print("resp")
#         print(resp)
#         if len(resp) < 8:
#             print(len(resp), resp)
#         command, nItems = struct.unpack("II", resp)

#         print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#         print(command)
#         if command == 3:
#             return 'end'
#         ret = {}
#         for i in range(nItems):
#             dr = self.s.recv(8 * 4)
#             eid, startTime, endTime, throughput = struct.unpack("Qddd", dr)
#             temp = {"startTime": startTime, "endTime": endTime, "throughput": throughput}
#             ret[eid] = temp
#         return ret


#     def disconnect(self):
#         # self.sendAsyncRequest(requestType=2, array=[])
#         self.s.close()

