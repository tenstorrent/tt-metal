// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// WebSocket client includes
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <server/websocket_test_client.hpp>

/**************************************************************************************************
| WebSocket Test Client
**************************************************************************************************/

typedef websocketpp::client<websocketpp::config::asio_client> client;
typedef websocketpp::config::asio_client::message_type::ptr message_ptr;
typedef websocketpp::connection_hdl connection_hdl;

class WebSocketTestClient {
private:
    client ws_client_;
    std::thread client_thread_;
    std::atomic<bool> running_{false};
    std::string uri_;
    connection_hdl connection_;
    std::mutex connection_mutex_;
    bool connected_{false};

    void on_open(connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        connection_ = hdl;
        connected_ = true;
        std::cout << "🎉 [Client] Connected to WebSocket server!" << std::endl;
    }

    void on_close(connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        connected_ = false;
        std::cout << "❌ [Client] Disconnected from WebSocket server" << std::endl;
    }

    void on_message(connection_hdl hdl, message_ptr msg) {
        std::cout << "📨 [Client] Received: " << msg->get_payload() << std::endl;

        // Send a response back every few messages (simple counter)
        static int message_count = 0;
        message_count++;

        if (message_count % 3 == 0) {
            std::string response = "Client response #" + std::to_string(message_count);
            try {
                ws_client_.send(hdl, response, websocketpp::frame::opcode::text);
                std::cout << "📤 [Client] Sent: " << response << std::endl;
            } catch (const websocketpp::exception& e) {
                std::cout << "❌ [Client] Failed to send response: " << e.what() << std::endl;
            }
        }
    }

    void on_fail(connection_hdl hdl) { std::cout << "❌ [Client] Connection failed" << std::endl; }

public:
    WebSocketTestClient(const std::string& uri) : uri_(uri) {
        // Disable all websocketpp logging for clean output
        ws_client_.clear_access_channels(websocketpp::log::alevel::all);
        ws_client_.clear_error_channels(websocketpp::log::elevel::all);

        // Initialize ASIO
        ws_client_.init_asio();

        // Set handlers
        ws_client_.set_open_handler([this](connection_hdl hdl) { on_open(hdl); });

        ws_client_.set_close_handler([this](connection_hdl hdl) { on_close(hdl); });

        ws_client_.set_message_handler([this](connection_hdl hdl, message_ptr msg) { on_message(hdl, msg); });

        ws_client_.set_fail_handler([this](connection_hdl hdl) { on_fail(hdl); });
    }

    void start() {
        running_ = true;
        client_thread_ = std::thread([this]() {
            try {
                std::cout << "🚀 [Client] Connecting to " << uri_ << "..." << std::endl;

                websocketpp::lib::error_code ec;
                client::connection_ptr con = ws_client_.get_connection(uri_, ec);

                if (ec) {
                    std::cout << "❌ [Client] Could not create connection: " << ec.message() << std::endl;
                    return;
                }

                ws_client_.connect(con);

                std::cout << "🔄 [Client] Starting event loop for 30 seconds..." << std::endl;

                // Start a timer thread to stop the client after 30 seconds
                std::thread timer_thread([this]() {
                    std::this_thread::sleep_for(std::chrono::seconds(30));
                    std::cout << "⏰ [Client] 30 second timer expired, stopping..." << std::endl;
                    running_ = false;
                    ws_client_.stop();
                });

                // Run the event loop until stopped
                ws_client_.run();

                // Wait for timer thread to finish
                if (timer_thread.joinable()) {
                    timer_thread.join();
                }

                // Close connection gracefully
                if (connected_) {
                    std::lock_guard<std::mutex> lock(connection_mutex_);
                    ws_client_.close(connection_, websocketpp::close::status::normal, "Client shutting down");
                }

            } catch (const websocketpp::exception& e) {
                std::cout << "❌ [Client] WebSocket exception: " << e.what() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "❌ [Client] Exception: " << e.what() << std::endl;
            }

            std::cout << "🏁 [Client] WebSocket test client finished" << std::endl;
        });
    }

    void stop() {
        running_ = false;
        if (client_thread_.joinable()) {
            client_thread_.join();
        }
    }

    ~WebSocketTestClient() { stop(); }
};

void run_websocket_test_client(uint16_t port) {
    std::cout << "🔌 Starting WebSocket test client..." << std::endl;

    // Give the server a moment to start up
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::string uri = "ws://localhost:" + std::to_string(port);
    WebSocketTestClient client(uri);
    client.start();

    // Wait for client to finish (it runs for 30 seconds)
    std::this_thread::sleep_for(std::chrono::seconds(32));
}
