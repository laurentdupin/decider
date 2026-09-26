#ifndef DECIDER_NATIVE_PROTOCOL_H
#define DECIDER_NATIVE_PROTOCOL_H

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace decider::native {

using Json = nlohmann::ordered_json;

enum class AnswerType { Choice, Score, Noul };

struct Question {
    std::string id;
    AnswerType type = AnswerType::Choice;
    std::string instructions;
    std::vector<std::string> option_names;
    std::vector<std::string> options;
    std::vector<std::string> legend;
    bool isolated = true;
};

struct Row {
    std::size_t question_index = 0;
    AnswerType temperature_type = AnswerType::Choice;
    std::string question;
    std::vector<std::string> options;
    bool isolated_level = false;
    std::size_t isolated_level_index = 0;
};

struct RequestPlan {
    std::string rendered_state;
    std::vector<Question> questions;
    std::vector<Row> rows;
    bool independent = true;
};

std::string python_json(const Json& value);
Json annotate_indices(const Json& value, std::size_t minimum_length = 8);
RequestPlan parse_system_one(std::string_view request_json, bool isolated_levels);

std::string plain_narrow_block(
    const Row& row, std::size_t answer_index, bool multi_question);

Json assemble_response(
    const RequestPlan& plan,
    const std::vector<std::vector<double>>& row_probabilities,
    std::string_view model_name,
    std::size_t input_tokens);

const char* answer_type_name(AnswerType type) noexcept;

}  // namespace decider::native

#endif
