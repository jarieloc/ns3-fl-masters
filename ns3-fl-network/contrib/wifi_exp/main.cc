#include <fcntl.h>
#include <unistd.h>
/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "fl-experiment.h"
#include "fl-sim-interface.h"
#include "ns3/simulator.h"
#include "ns3/log.h"

#include <chrono>
#include <memory>
#include <random>
#include <ctime>
#include <map>

using sysclock_t = std::chrono::system_clock;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Wifi-Adhoc");

// Global socket provider on fixed port 8080 (matches Python)
FLSimProvider g_fLSimProvider(8080);
std::map<int, std::shared_ptr<ClientSession>> g_clients;

// External toggle used inside fl-sim-interface.cc
extern bool g_use_fl_socket;

int main(int argc, char* argv[])
{
    // ---- CLI defaults ----
    bool UseFLSocket   = false;
    double SimTime     = 10.0;
    std::string dataRate = "250kbps";
    int numClients     = 20;
    std::string NetworkType = "wifi";
    int MaxPacketSize  = 1024;     // bytes
    double TxGain      = 0.0;      // dB (+30 => dBm)
    double ModelSizeKB = 15.00;    // kilobytes, will be passed through to Experiment
    std::string learningModel = "sync"; // "sync" or "async"

    CommandLine cmd(__FILE__);
    cmd.AddValue("UseFLSocket", "Wait for external FL socket", UseFLSocket);
    cmd.AddValue("SimTime",     "Stop the simulator after this many seconds", SimTime);
    cmd.AddValue("NumClients",  "Number of clients", numClients);
    cmd.AddValue("NetworkType", "Type of network", NetworkType);
    cmd.AddValue("MaxPacketSize","Maximum size packet that can be sent", MaxPacketSize);
    cmd.AddValue("TxGain",      "Power transmitted from clients and server", TxGain);
    cmd.AddValue("ModelSize",   "Size of model (KB)", ModelSizeKB);
    cmd.AddValue("DataRate",    "Application data rate", dataRate);
    cmd.AddValue("LearningModel","Async or Sync federated learning", learningModel);
    cmd.Parse(argc, argv);

    // Wire the global toggle used in fl-sim-interface.*
    g_use_fl_socket = UseFLSocket;

    // Quiet stdin when NOT using the socket (prevents accidental blocking on read)
    if (!UseFLSocket) {
        int _fd = open("/dev/null", O_RDONLY);
        if (_fd >= 0) { dup2(_fd, 0); close(_fd); }
    }

    const bool bAsync = (learningModel == "async");

    // Minimal banner
    NS_LOG_UNCOND("{NumClients:" << numClients
                   << ",NetworkType:" << NetworkType
                   << ",MaxPacketSize:" << MaxPacketSize
                   << ",TxGain:" << TxGain << "}");

    // Timestamped CSV (kept identical to original)
    std::time_t now = sysclock_t::to_time_t(sysclock_t::now());
    char buf[80] = {0};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%H-%S.csv", std::localtime(&now));
    char strBuff[100];
    snprintf(strBuff, 99, "%s_%s_%.2f_%s", learningModel.c_str(), NetworkType.c_str(), TxGain, buf);
    FILE* fp = fopen(strBuff, "w");

    // Initialize all client sessions with deterministic radii/angles
    for (int j = 0; j < numClients; j++) {
        double radius = static_cast<double>(5 << (j % 4 + 2)); // {20,40,80,160} repeating
        double theta  = (1.0 / numClients) * j;
        NS_LOG_UNCOND("INIT:J=" << j << " r=" << radius << " th=" << theta);
        g_clients[j] = std::make_shared<ClientSession>(j, radius, theta);
    }

    // If socket is enabled, start listening (bind+listen); accept happens inside recv()
    if (UseFLSocket) {
        NS_LOG_UNCOND("wifi_exp: UseFLSocket=true; waiting for connection on port 8080");
        g_fLSimProvider.waitForConnection();
    }

    ns3::Time timeOffset(0);
    int round = 0;

    while (true) {
        round++;

        // When using the socket, wait for Python's RUN_SIMULATION bitmap each round
        if (UseFLSocket) {
            FLSimProvider::COMMAND::Type type = g_fLSimProvider.recv(g_clients);
            if (type == FLSimProvider::COMMAND::Type::EXIT) {
                NS_LOG_UNCOND("wifi_exp: EXIT received; closing socket");
                g_fLSimProvider.Close();
                break;
            }
            // (If invalid/ignored commands are returned as RESPONSE by recv(), we still proceed)
        }

        // Build & run the experiment for this round
        auto experiment = Experiment(numClients,
                                     NetworkType,
                                     MaxPacketSize,
                                     TxGain,
                                     /* ModelSize */ ModelSizeKB,
                                     dataRate,
                                     bAsync,
                                     /* flSimProvider ptr */ &g_fLSimProvider,
                                     fp,
                                     round);
        // Produces { clientId -> Message{id, roundTime, throughput} }
        auto roundStats = experiment.WeakNetwork(g_clients, timeOffset);
        if (UseFLSocket) {
        g_fLSimProvider.send(roundStats);   // ALWAYS send when using the socket controller
    }

        // End time (seconds) after the run completed
        double simEnd = ns3::Simulator::Now().GetSeconds();

        // In non-socket mode, keep the legacy JSON summary (used by inline/THz-style consumers)
        if (!UseFLSocket) {
            const long long rxBytes = static_cast<long long>(ModelSizeKB * 1000.0); // KB->bytes (as in original)
            std::cout << "{\"clientResults\":[";
            for (int i = 0; i < numClients; ++i) {
                std::cout << "{\"id\":" << i
                          << ",\"rxBytes\":" << rxBytes
                          << ",\"doneAt\":" << simEnd
                          << "}";
                if (i + 1 < numClients) std::cout << ",";
            }
            std::cout << "]}" << std::endl;
            fflush(fp);
            // When not using the FL socket, do one round then exit (original behavior)
            break;
        }

        // Socket mode: for SYNC learning, *always* emit the RESPONSE now.
        if (!bAsync) {
            NS_LOG_UNCOND("wifi_exp: sending " << roundStats.size() << " result items to FL socket");
            g_fLSimProvider.send(roundStats);  // writes <II> + N * Message (binary)
        } else {
            // (If you later support async, call g_fLSimProvider.send(AsyncMessage*) per client)
            NS_LOG_UNCOND("wifi_exp: async model not wired for socket send in this fork");
        }

        NS_LOG_UNCOND(">>>>>>>>>>>>>>>>>>>>>>>>>\nTIME_OFFSET:" << timeOffset << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        fflush(fp);
    }

    fclose(fp);
    NS_LOG_UNCOND("Exiting c++");
    return 0;
}


// #include <fcntl.h>
// #include <unistd.h>
// /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
// /*
//  * Copyright (c) 2022 Emily Ekaireb
//  *
//  * This program is free software; you can redistribute it and/or modify
//  * it under the terms of the GNU General Public License version 2 as
//  * published by the Free Software Foundation;
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU General Public License for more details.
//  *
//  * You should have received a copy of the GNU General Public License
//  * along with this program; if not, write to the Free Software
//  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//  *
//  * Author: Emily Ekaireb <eekaireb@ucsd.edu>
//  */

// #include "fl-experiment.h"
// #include "ns3/simulator.h"

// #include <chrono>
// #include <memory>
// #include <random>

// using sysclock_t = std::chrono::system_clock;

// using namespace ns3;
// FLSimProvider g_fLSimProvider(8080);
// std::map<int, std::shared_ptr<ClientSession>> g_clients;

// NS_LOG_COMPONENT_DEFINE("Wifi-Adhoc");

// int
// main(int argc, char* argv[])
// {
//       bool UseFLSocket=false; double SimTime=10.0;
// // LogComponentEnable("PropagationLossModel", LOG_LEVEL_ALL);

//     FLSimProvider* flSimProvider = &g_fLSimProvider;

//     std::string dataRate = "250kbps"; /* Application layer data rate. */
//     int numClients = 20; // when numClients is 50 or greater, packets are not received by server
//     std::string NetworkType = "wifi";
//     int MaxPacketSize = 1024;      // bytes
//     double TxGain = 0.0;           // dB + 30 = dBm
//     double ModelSize = 1.500 * 10; // kb
//     std::string learningModel = "sync";

//     CommandLine cmd(__FILE__);

    
//   cmd.AddValue("UseFLSocket","Wait for external FL socket", UseFLSocket);
//   cmd.AddValue("SimTime","Stop the simulator after this many seconds", SimTime);
// cmd.AddValue("NumClients", "Number of clients", numClients);
//     cmd.AddValue("NetworkType", "Type of network", NetworkType);
//     cmd.AddValue("MaxPacketSize", "Maximum size packet that can be sent", MaxPacketSize);
//     cmd.AddValue("TxGain", "Power transmitted from clients and server", TxGain);
//     cmd.AddValue("ModelSize", "Size of model", ModelSize);
//     cmd.AddValue("DataRate", "Application data rate", dataRate);
//     cmd.AddValue("LearningModel", "Async or Sync federated learning", learningModel);

//     cmd.Parse(argc, argv);
//   extern bool g_use_fl_socket; g_use_fl_socket = UseFLSocket;
//   if (!UseFLSocket) {
//     int _fd = open("/dev/null", O_RDONLY);
//     if (_fd >= 0) { dup2(_fd, 0); close(_fd); }
//   }
//   extern bool g_use_fl_socket; g_use_fl_socket = UseFLSocket;
//   if (!UseFLSocket) {
//     int _fd = open("/dev/null", O_RDONLY);
//     if (_fd >= 0) { dup2(_fd, 0); close(_fd); }
//   }


//     bool bAsync = false;
//     if (learningModel == "async")
//     {
//         bAsync = true;
//     }

//     // ModelSize = ModelSize * 1000; // conversion to bytes

//     NS_LOG_UNCOND(
//             "{NumClients:" << numClients << ","
//             "NetworkType:" << NetworkType << ","
//             "MaxPacketSize:" << MaxPacketSize << ","
//             "TxGain:" << TxGain << "}"
//     );

//     //Experiment experiment(numClients,NetworkType,MaxPacketSize,TxGain);

//     std::time_t now = sysclock_t::to_time_t(sysclock_t::now());

//     char buf[80] = {0};
//     std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%H-%S.csv", std::localtime(&now));

//     char strBuff[100];
//     snprintf(strBuff, 99, "%s_%s_%.2f_%s", learningModel.c_str(), NetworkType.c_str(), TxGain, buf);

//     FILE* fp = fopen(strBuff, "w");

//     std::default_random_engine generator;
//     std::uniform_real_distribution<double> r_dist(1.0, 4.0);
//     // std::uniform_real_distribution<double> t_dist(0,1.0);

//     // initialize structure for all clients
//     for (int j = 0; j < numClients; j++)
//     {
//         // place the nodes at random spots from the base station

//         double radius = (double)(5 << (j % 4 + 2));
//         // double theta = t_dist(generator);
//         double theta = (1.0 / numClients) * (j);

//         NS_LOG_UNCOND("INIT:J=" << j << " r=" << radius << " th=" << theta);
//         g_clients[j] = std::make_shared<ClientSession>(j, radius, theta);
//     }

//     ns3::Time timeOffset(0);

//     if (flSimProvider)
//     {
//         g_fLSimProvider.waitForConnection();
//     }

//     int round = 0;

//     while (true)
//     {
//         round++;

//         if (flSimProvider)
//         {
//             FLSimProvider::COMMAND::Type type = g_fLSimProvider.recv(g_clients);

//             if (type == FLSimProvider::COMMAND::Type::EXIT)
//             {
//                 g_fLSimProvider.Close();
//                 break;
//             }
//         }

//         auto experiment = Experiment(numClients,
//                                      NetworkType,
//                                      MaxPacketSize,
//                                      TxGain,
//                                      ModelSize,
//                                      dataRate,
//                                      bAsync,
//                                      flSimProvider,
//                                      fp,
//                                      round

//         );
//         auto roundStats = experiment.WeakNetwork(g_clients, timeOffset);

//         // End time (seconds) after the run completed
//         double simEnd = ns3::Simulator::Now().GetSeconds();

//         // Minimal JSON block the Python expects
//         std::cout << "{\"clientResults\":[";
//         for (int i = 0; i < numClients; ++i) {
//         std::cout << "{\"id\":" << i
//                     << ",\"rxBytes\":" << (long long)ModelSize
//                     << ",\"doneAt\":" << simEnd
//                     << "}";
//         if (i + 1 < numClients) std::cout << ",";
//         }
//         std::cout << "]}" << std::endl;

//         // If not using the external FL socket, do one round and exit cleanly.
//         if (!UseFLSocket) { fflush(fp); break; }

//         NS_LOG_UNCOND(
//             ">>>>>>>>>>>>>>>>>>>>>>>>>\nTIME_OFFSET:"
//             << timeOffset << "\n"
//             ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
//         );

//         if (flSimProvider && !bAsync)
//         {
//             g_fLSimProvider.send(roundStats);
//         }
//         if (!flSimProvider)
//         {
//             break;
//         }

//         fflush(fp);
//     }

//     fclose(fp);
//     NS_LOG_UNCOND("Exiting c++");

//     return 0;
// }
