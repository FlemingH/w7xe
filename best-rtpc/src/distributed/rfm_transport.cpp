#include "distributed/rfm_transport.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace best_rtpc {

RfmTransport::RfmTransport()
    : sock_fd_(-1), conn_fd_(-1), is_sender_(false) {}

RfmTransport::~RfmTransport() {
    if (conn_fd_ >= 0) close(conn_fd_);
    if (sock_fd_ >= 0) close(sock_fd_);
}

void RfmTransport::init_sender(int port) {
    is_sender_ = true;
    sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ < 0) { perror("socket"); exit(1); }

    int opt = 1;
    setsockopt(sock_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(sock_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); exit(1);
    }
    listen(sock_fd_, 1);

    printf("[RFM-TX] Listening on port %d...\n", port);
    conn_fd_ = accept(sock_fd_, nullptr, nullptr);
    if (conn_fd_ < 0) { perror("accept"); exit(1); }
    printf("[RFM-TX] Connected.\n");
}

void RfmTransport::init_receiver(int port) {
    is_sender_ = false;
    conn_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (conn_fd_ < 0) { perror("socket"); exit(1); }

    int opt = 1;
    setsockopt(conn_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(port);

    printf("[RFM-RX] Connecting to port %d...\n", port);
    while (connect(conn_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        usleep(100000);  // retry every 100ms
    }
    printf("[RFM-RX] Connected.\n");
}

void RfmTransport::send_raw(const void* data, size_t bytes) {
    const char* ptr = (const char*)data;
    size_t sent = 0;
    while (sent < bytes) {
        ssize_t n = write(conn_fd_, ptr + sent, bytes - sent);
        if (n <= 0) { perror("send_raw"); exit(1); }
        sent += n;
    }
}

void RfmTransport::recv_raw(void* data, size_t bytes) {
    char* ptr = (char*)data;
    size_t recvd = 0;
    while (recvd < bytes) {
        ssize_t n = read(conn_fd_, ptr + recvd, bytes - recvd);
        if (n <= 0) { perror("recv_raw"); exit(1); }
        recvd += n;
    }
}

void RfmTransport::send_equilibrium(const EquilibriumData& eq) {
    // Header: grid dimensions + scalar parameters
    int header[2] = {eq.nr, eq.nz};
    send_raw(header, sizeof(header));

    float scalars[8] = {
        eq.R_min, eq.R_max, eq.Z_min, eq.Z_max,
        eq.psi_axis, eq.psi_boundary, eq.R_axis, eq.Z_axis
    };
    send_raw(scalars, sizeof(scalars));

    // Array data
    size_t n = (size_t)eq.nr * eq.nz;
    send_raw(eq.psi,  n * sizeof(float));
    send_raw(eq.ne,   n * sizeof(float));
    send_raw(eq.Te,   n * sizeof(float));
    send_raw(eq.Bphi, n * sizeof(float));
}

void RfmTransport::receive_equilibrium(EquilibriumData& eq) {
    int header[2];
    recv_raw(header, sizeof(header));
    eq.nr = header[0];
    eq.nz = header[1];

    float scalars[8];
    recv_raw(scalars, sizeof(scalars));
    eq.R_min = scalars[0]; eq.R_max = scalars[1];
    eq.Z_min = scalars[2]; eq.Z_max = scalars[3];
    eq.psi_axis = scalars[4]; eq.psi_boundary = scalars[5];
    eq.R_axis = scalars[6]; eq.Z_axis = scalars[7];

    size_t n = (size_t)eq.nr * eq.nz;
    eq.psi  = new float[n];
    eq.ne   = new float[n];
    eq.Te   = new float[n];
    eq.Bphi = new float[n];
    eq.BR   = nullptr;
    eq.BZ   = nullptr;

    recv_raw(eq.psi,  n * sizeof(float));
    recv_raw(eq.ne,   n * sizeof(float));
    recv_raw(eq.Te,   n * sizeof(float));
    recv_raw(eq.Bphi, n * sizeof(float));
}

void RfmTransport::local_transfer(const EquilibriumData& src,
                                   EquilibriumData& dst) {
    dst.nr = src.nr; dst.nz = src.nz;
    dst.R_min = src.R_min; dst.R_max = src.R_max;
    dst.Z_min = src.Z_min; dst.Z_max = src.Z_max;
    dst.psi_axis = src.psi_axis; dst.psi_boundary = src.psi_boundary;
    dst.R_axis = src.R_axis; dst.Z_axis = src.Z_axis;

    size_t n = (size_t)src.nr * src.nz;
    dst.psi  = new float[n];
    dst.ne   = new float[n];
    dst.Te   = new float[n];
    dst.Bphi = new float[n];
    dst.BR   = nullptr;
    dst.BZ   = nullptr;

    std::memcpy(dst.psi,  src.psi,  n * sizeof(float));
    std::memcpy(dst.ne,   src.ne,   n * sizeof(float));
    std::memcpy(dst.Te,   src.Te,   n * sizeof(float));
    std::memcpy(dst.Bphi, src.Bphi, n * sizeof(float));
}

}  // namespace best_rtpc
