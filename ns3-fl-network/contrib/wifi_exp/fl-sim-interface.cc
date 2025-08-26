// Optional external FL socket toggle
bool g_use_fl_socket = false;

/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2022 Emily Ekaireb
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Emily Ekaireb <eekaireb@ucsd.edu>
 */
#include "fl-sim-interface.h"

namespace ns3
{

void
FLSimProvider::waitForConnection()
{
    if (!g_use_fl_socket) return;

    // Create TCP socket
    m_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (m_server_fd < 0) {
        NS_LOG_UNCOND("FLSimProvider: socket() failed, errno=" << errno);
        return; // don't kill the whole process
    }

    // Reuse (avoid "address already in use" on quick restarts)
    int opt = 1;
    setsockopt(m_server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    // Bind to all interfaces on the configured port
    memset(&m_address, 0, sizeof(m_address));
    m_address.sin_family = AF_INET;
    m_address.sin_addr.s_addr = INADDR_ANY;
    m_address.sin_port = htons(m_port);

    if (bind(m_server_fd, (struct sockaddr*)&m_address, sizeof(m_address)) < 0) {
        NS_LOG_UNCOND("FLSimProvider: bind(" << m_port << ") FAILED, errno=" << errno);
        // Keep going; recv() will keep trying to accept when we recover
    }

    if (listen(m_server_fd, 3) < 0) {
        NS_LOG_UNCOND("FLSimProvider: listen FAILED, errno=" << errno);
        // Keep going to avoid hard exit
    }

    NS_LOG_UNCOND("FLSimProvider: listening on 0.0.0.0:" << m_port);

    // DO NOT accept here. Accept happens inside recv()'s loop.
    m_new_socket = -1;
}

// FLSimProvider::COMMAND::Type
// FLSimProvider::recv(std::map<int, std::shared_ptr<ClientSession>>& packetsReceived)
// {
//     COMMAND c;
//     int len = read(m_new_socket, (char*)&c, sizeof(c));

//     if (len != sizeof(COMMAND))
//     {
//         if (len != 0)
//         {
//             NS_LOG_UNCOND("Invalid Command: Len(" << len << ")!=(" << sizeof(COMMAND) << ")");
//         }
//         else
//         {
//             NS_LOG_UNCOND("Socket closed by Python");
//         }
//         close(m_new_socket);
//         return COMMAND::Type::EXIT;
//     }

//     if (c.command == COMMAND::Type::EXIT)
//     {
//         NS_LOG_UNCOND("Exit Called");
//         close(m_new_socket);
//         return COMMAND::Type::EXIT;
//     }
//     else if (c.command != COMMAND::Type::RUN_SIMULATION)
//     {
//         NS_LOG_UNCOND("Invalid command");
//         close(m_new_socket);
//         return COMMAND::Type::EXIT;
//     }
//     else if (packetsReceived.size() != c.nItems)
//     {
//         NS_LOG_UNCOND("Invalid number of clients");
//         close(m_new_socket);
//         return COMMAND::Type::EXIT;
//     }

FLSimProvider::COMMAND::Type
FLSimProvider::recv(std::map<int, std::shared_ptr<ClientSession>>& packetsReceived)
{
    COMMAND c;

    for (;;)
    {
        if (m_new_socket <= 0)
        {
            int addrlen = sizeof(m_address);
            m_new_socket = accept(m_server_fd, (struct sockaddr*)&m_address, (socklen_t*)&addrlen);
            if (m_new_socket < 0)
            {
                // keep waiting; do not exit the whole app
                continue;
            }
        }

        int len = read(m_new_socket, (char*)&c, sizeof(c));

        if (len == sizeof(COMMAND))
        {
            break; // got a full header; handle below
        }

        if (len == 0)
        {
            NS_LOG_UNCOND("Peer disconnected before sending header; waiting...");
            close(m_new_socket);
            m_new_socket = -1;
            continue; // accept next peer
        }

        // Partial/garbage; drop connection and keep listening
        NS_LOG_UNCOND("Invalid Command: Len(" << len << ")!=(" << sizeof(COMMAND) << ")");
        close(m_new_socket);
        m_new_socket = -1;
        continue;
    }

    if (c.command == COMMAND::Type::EXIT)
    {
        NS_LOG_UNCOND("Exit Called");
        close(m_new_socket);
        m_new_socket = -1;
        return COMMAND::Type::EXIT;
    }

    if (c.command != COMMAND::Type::RUN_SIMULATION)
    {
        NS_LOG_UNCOND("Invalid command value");
        close(m_new_socket);
        m_new_socket = -1;
        return COMMAND::Type::RESPONSE; // ignore and keep server alive
    }

    // Expect a bitmap of size == total clients
    if (packetsReceived.size() != c.nItems)
    {
        NS_LOG_UNCOND("Invalid number of clients; got " << packetsReceived.size() << " expected " << c.nItems);
        close(m_new_socket);
        m_new_socket = -1;
        return COMMAND::Type::RESPONSE;
    }

    int i = 0;
    for (auto it = packetsReceived.begin(); it != packetsReceived.end(); it++, i++)
    {
        uint32_t temp;

        if (sizeof(temp) != read(m_new_socket, (char*)&temp, sizeof(temp)))
        {
            NS_LOG_UNCOND("Invalid valid length received");
            return COMMAND::Type::EXIT;
        }

        it->second->SetInRound(temp != 0);
    }

    NS_LOG_UNCOND("wifi_exp: RUN_SIMULATION received; nItems=" << c.nItems);
    return c.command;
}

void
FLSimProvider::Close()
{
   if (!g_use_fl_socket) return;
 if (!g_use_fl_socket) return;
close(m_new_socket);
}

void
FLSimProvider::send(AsyncMessage* pMessage)
{
   if (!g_use_fl_socket) return;
 if (!g_use_fl_socket) return;
COMMAND r;
    r.command = COMMAND::Type::RESPONSE;
    r.nItems = 1;
    write(m_new_socket, (char*)&r, sizeof(r));
    write(m_new_socket, pMessage, sizeof(AsyncMessage));
}

void
FLSimProvider::end()
{
   if (!g_use_fl_socket) return;
 if (!g_use_fl_socket) return;
COMMAND r;
    r.command = COMMAND::Type::ENDSIM;
    r.nItems = 0;
    write(m_new_socket, (char*)&r, sizeof(r));
}

void
FLSimProvider::send(std::map<int, Message>& roundTime)
{
    if (!g_use_fl_socket) return;
    if (m_new_socket < 0) return;

    COMMAND r;
    r.command = COMMAND::Type::RESPONSE;
    r.nItems  = roundTime.size();

    ssize_t w = write(m_new_socket, (char*)&r, sizeof(r));
    if (w != (ssize_t)sizeof(r)) { close(m_new_socket); m_new_socket = -1; return; }

    for (auto it = roundTime.begin(); it != roundTime.end(); ++it)
    {
        Message msg = it->second;   // make a copy
        msg.id = it->first;         // set the id we’re returning

        w = write(m_new_socket, (char*)&msg, sizeof(Message));
        if (w != (ssize_t)sizeof(Message)) { close(m_new_socket); m_new_socket = -1; return; }
    }

    roundTime.clear();
}



} // namespace ns3
