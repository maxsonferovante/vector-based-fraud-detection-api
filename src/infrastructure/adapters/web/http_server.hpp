#ifndef HTTP_SERVER_HPP
#define HTTP_SERVER_HPP

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <memory>
#include "web_adapter.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

namespace infrastructure {
namespace adapters {
namespace web {

class Session : public std::enable_shared_from_this<Session> {
    tcp::socket socket_;
    beast::flat_buffer buffer_;
    http::request<http::string_body> req_;
    std::shared_ptr<WebAdapter> adapter_;

public:
    Session(tcp::socket socket, std::shared_ptr<WebAdapter> adapter)
        : socket_(std::move(socket)), adapter_(std::move(adapter)) {}

    void run() { do_read(); }

private:
    void do_read() {
        auto self = shared_from_this();
        http::async_read(socket_, buffer_, req_, [this, self](beast::error_code ec, std::size_t) {
            if (!ec) {
                beast::error_code ec_ep;
                auto ep = socket_.remote_endpoint(ec_ep);
                do_write(adapter_->handle_request(ep, req_));
            } else if (ec == http::error::end_of_stream) {
                beast::error_code ignored;
                socket_.shutdown(tcp::socket::shutdown_send, ignored);
            }
        });
    }

    void do_write(http::response<http::string_body> res) {
        auto self = shared_from_this();
        auto sp = std::make_shared<http::response<http::string_body>>(std::move(res));
        http::async_write(socket_, *sp, [this, self, sp](beast::error_code ec, std::size_t) {
            if (!ec && sp->keep_alive()) {
                req_ = {};
                buffer_.consume(buffer_.size());
                do_read();
            } else {
                beast::error_code ignored;
                socket_.shutdown(tcp::socket::shutdown_send, ignored);
            }
        });
    }
};

class Listener : public std::enable_shared_from_this<Listener> {
    net::io_context& ioc_;
    tcp::acceptor acceptor_;
    std::shared_ptr<WebAdapter> adapter_;

public:
    Listener(net::io_context& ioc, tcp::endpoint endpoint, std::shared_ptr<WebAdapter> adapter)
        : ioc_(ioc), acceptor_(net::make_strand(ioc)), adapter_(std::move(adapter)) {
        beast::error_code ec;
        acceptor_.open(endpoint.protocol(), ec);
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        acceptor_.bind(endpoint, ec);
        acceptor_.listen(net::socket_base::max_listen_connections, ec);
    }

    void run() { do_accept(); }

private:
    void do_accept() {
        auto self = shared_from_this();
        acceptor_.async_accept(net::make_strand(ioc_), [self](beast::error_code ec, tcp::socket socket) {
            if (!ec) std::make_shared<Session>(std::move(socket), self->adapter_)->run();
            self->do_accept();
        });
    }
};

} // namespace web
} // namespace adapters
} // namespace infrastructure

#endif
