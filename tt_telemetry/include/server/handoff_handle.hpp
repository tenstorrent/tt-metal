#pragma once

/*
 * handoff_handle.hpp
 *
 * Object designed for handing off a container from a producer to a consumer. A release callback
 * notifies the producer of when the consumer has released the handle. There is no explicit thread-
 * safe synchronization here; the producer is expected to implement that.
 */

#include <memory>
#include <functional>

template <typename Container>
class HandoffHandle {
private:
    std::shared_ptr<Container> container_;
    std::function<void(std::shared_ptr<Container>)> release_callback_;

public:
    HandoffHandle(std::shared_ptr<Container> buffer, std::function<void(std::shared_ptr<Container>)> release_callback)
        : container_(buffer)
        , release_callback_(release_callback) {
    }

    HandoffHandle(HandoffHandle<Container> &&other) noexcept
        : container_(std::move(other.container_))
        , release_callback_(std::move(other.release_callback_)) {
        other.container_ = nullptr;
    }

    HandoffHandle& operator=(HandoffHandle<Container> &&other) noexcept {
        if (this != &other) {
            if (container_ && release_callback_) {
                release_callback_(container_);
            }
            container_ = std::move(other.container_);
            release_callback_ = std::move(other.release_callback_);
            other.container_ = nullptr;
        }
        return *this;
    }

    HandoffHandle(const HandoffHandle<Container> &) = delete;
    HandoffHandle &operator=(const HandoffHandle<Container> &) = delete;

    ~HandoffHandle() {
        if (container_ && release_callback_) {
            release_callback_(container_);
        }
    }

    Container *operator->() {
        return container_.get();
    }

    Container &operator*() {
        return *container_;
    }

    const Container *operator->() const {
        return container_.get();
    }

    const Container &operator*() const {
        return *container_;
    }

    bool is_valid() const {
        return container_ != nullptr;
    }
};