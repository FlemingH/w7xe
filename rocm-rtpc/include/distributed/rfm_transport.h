#pragma once

#include "common/types.h"

namespace rocm_rtpc {

// Simulated Reflective Memory (RFM) transport layer.
// In production, this would use an actual RFM network card (e.g., GE VMIPCI-5565)
// for deterministic low-latency data transfer between PCS and ECRH systems.
//
// This implementation uses TCP sockets to simulate the RFM data path,
// or local shared memory for single-machine testing.
class RfmTransport {
public:
    RfmTransport();
    ~RfmTransport();

    // PCS side: initialize as data sender
    void init_sender(int port);

    // ECRH side: initialize as data receiver
    void init_receiver(int port);

    // Send equilibrium data (PCS → ECRH)
    void send_equilibrium(const EquilibriumData& eq);

    // Receive equilibrium data (ECRH ← PCS)
    void receive_equilibrium(EquilibriumData& eq);

    // Local mode: direct memory copy (for single-machine e2e testing)
    static void local_transfer(const EquilibriumData& src,
                               EquilibriumData& dst);

private:
    int sock_fd_;
    int conn_fd_;
    bool is_sender_;

    void send_raw(const void* data, size_t bytes);
    void recv_raw(void* data, size_t bytes);
};

}  // namespace rocm_rtpc
