#pragma once

/*
 * server/json_messages.hpp
 *
 * Defines JSON messages that the server uses to communicate with the frontend.
 */

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace messages {
    struct EndpointDescription {
        std::string from;
        std::string to;
    };

    void to_json(nlohmann::json &j, const EndpointDescription &d) {
        j = nlohmann::json{{"from", d.from}, {"to", d.to}};
    }

    void from_json(const nlohmann::json &j, EndpointDescription &d) {
        j.at("from").get_to(d.from);
        j.at("to").get_to(d.to);
    }

    struct EndpointDefinitionMessage {
        const char *type = "EndpointDefinitionMessage";
        std::vector<EndpointDescription> endpoints;
    };

    void to_json(nlohmann::json &j, const EndpointDefinitionMessage &e) {
        j = nlohmann::json{{"type", e.type}, {"endpoints", e.endpoints}};
    }

    void from_json(const nlohmann::json& j, EndpointDefinitionMessage &e) {
        j.at("endpoints").get_to(e.endpoints);
    }
}