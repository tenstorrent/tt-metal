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
    // Describes an endpoint. "from" is the name of the endpoint, "to" is the name of the end it
    // is connected to.
    struct EndpointDescription {
        size_t id;              // unique id
        std::string from;
        std::string to;
        uint8_t state;
    };

    static inline void to_json(nlohmann::json &j, const EndpointDescription &d) {
        j = nlohmann::json {
            { "id", d.id },
            { "from", d.from }, 
            { "to", d.to },
            { "state", d.state }
        };
    }

    static inline void from_json(const nlohmann::json &j, EndpointDescription &d) {
        j.at("id").get_to(d.id);
        j.at("from").get_to(d.from);
        j.at("to").get_to(d.to);
        j.at("state").get_to(d.state);
    }

    // Transmits descriptions of newly detected endpoints
    struct EndpointDefinitionMessage {
        const char *type = "EndpointDefinitionMessage";
        std::string host;
        std::vector<EndpointDescription> endpoints;
    };

    static inline void to_json(nlohmann::json &j, const EndpointDefinitionMessage &e) {
        j = nlohmann::json{
            { "type", e.type },
            { "host", e.host },
            { "endpoints", e.endpoints }
        };
    }

    static inline void from_json(const nlohmann::json &j, EndpointDefinitionMessage &e) {
        j.at("host").get_to(e.host);
        j.at("endpoints").get_to(e.endpoints);
    }

    // Transmits updates for endpoints whose state has changed
    struct EndpointStateChangeMessage {
        const char *type = "EndpointStateChangeMessage";
        std::string host;
        std::vector<size_t> endpoint_indices;
        std::vector<uint8_t> endpoint_states;
    };

    static inline void to_json(nlohmann::json &j, const EndpointStateChangeMessage &s) {
        j = nlohmann::json({
            { "type", s.type },
            { "host", s.host },
            { "endpoint_indices", s.endpoint_indices },
            { "endpoint_states", s.endpoint_states }
        });
    }

    static inline void from_json(const nlohmann::json &j, EndpointStateChangeMessage &s) {
        j.at("host").get_to(s.host);
        j.at("endpoint_indices").get_to(s.endpoint_indices);
        j.at("endpoint_states").get_to(s.endpoint_states);
    }
}