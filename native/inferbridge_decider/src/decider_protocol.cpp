#include "decider_protocol.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace decider::native {
namespace {

constexpr std::size_t kMaximumChoiceOptions = 255;
constexpr std::size_t kMaximumScoreLevels = 10;
constexpr char kLetters[] = "ABCDEFGHIJ";
constexpr std::string_view kNoulWithoutInstructions =
    "Which answer fits the context?";
constexpr std::string_view kIsolatedTemplatePrefix = "\nProposed answer: ";
constexpr std::string_view kIsolatedTemplateSuffix =
    "\nDoes the proposed answer fit?";

std::string scalar_json(const Json& value) {
    return value.dump(-1, ' ', false, Json::error_handler_t::strict);
}

void append_python_json(std::string& output, const Json& value) {
    if (value.is_array()) {
        output.push_back('[');
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) output += ", ";
            append_python_json(output, value[index]);
        }
        output.push_back(']');
        return;
    }
    if (value.is_object()) {
        output.push_back('{');
        bool first = true;
        for (const auto& item : value.items()) {
            if (!first) output += ", ";
            first = false;
            output += Json(item.key()).dump(-1, ' ', false, Json::error_handler_t::strict);
            output += ": ";
            append_python_json(output, item.value());
        }
        output.push_back('}');
        return;
    }
    output += scalar_json(value);
}

std::string text(const Json& value) {
    return value.is_string() ? value.get<std::string>() : python_json(value);
}

std::string python_repr(const Json& value) {
    if (value.is_null()) return "None";
    if (value.is_boolean()) return value.get<bool>() ? "True" : "False";
    if (value.is_string()) {
        std::string output = "'";
        for (const unsigned char character : value.get_ref<const std::string&>()) {
            switch (character) {
                case '\\': output += "\\\\"; break;
                case '\'': output += "\\'"; break;
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                default: output.push_back(static_cast<char>(character)); break;
            }
        }
        return output + "'";
    }
    if (value.is_array()) {
        std::string output = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) output += ", ";
            output += python_repr(value[index]);
        }
        return output + "]";
    }
    if (value.is_object()) {
        std::string output = "{";
        bool first = true;
        for (const auto& item : value.items()) {
            if (!first) output += ", ";
            first = false;
            output += python_repr(Json(item.key())) + ": " + python_repr(item.value());
        }
        return output + "}";
    }
    return scalar_json(value);
}

std::string python_str(const Json& value) {
    return value.is_string() ? value.get<std::string>() : python_repr(value);
}

bool missing_description(const Json& value) {
    return value.is_null() || (value.is_string() && value.get_ref<const std::string&>().empty());
}

const Json* optional_member(const Json& value, std::string_view name) {
    if (!value.is_object()) return nullptr;
    const auto found = value.find(std::string(name));
    return found == value.end() ? nullptr : &*found;
}

std::string strip_level_number(std::string value) {
    std::size_t position = 0;
    while (position < value.size() && std::isspace(static_cast<unsigned char>(value[position]))) ++position;
    if (position < value.size() && value[position] == '-') ++position;
    const std::size_t digits = position;
    while (position < value.size() && std::isdigit(static_cast<unsigned char>(value[position]))) ++position;
    if (position == digits) return value;
    while (position < value.size() && std::isspace(static_cast<unsigned char>(value[position]))) ++position;
    if (position >= value.size() || value[position] != ':') return value;
    ++position;
    while (position < value.size() && std::isspace(static_cast<unsigned char>(value[position]))) ++position;
    return value.substr(position);
}

double clip01(double value) {
    return std::min(1.0, std::max(0.0, value));
}

std::vector<double> normalized(std::vector<double> probabilities) {
    if (probabilities.empty()) throw std::invalid_argument("an answer has no probabilities");
    for (const double value : probabilities) {
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument("answer probabilities must be finite and non-negative");
    }
    const double total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    if (total == 0.0) {
        std::fill(probabilities.begin(), probabilities.end(), 1.0 / probabilities.size());
    } else {
        for (double& value : probabilities) value /= total;
    }
    return probabilities;
}

std::vector<double> format_normalized(std::vector<double> probabilities) {
    if (probabilities.empty()) throw std::invalid_argument("an answer has no probabilities");
    for (const double value : probabilities) {
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument("answer probabilities must be finite and non-negative");
    }
    const double total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    if (total != 0.0)
        for (double& value : probabilities) value /= total;
    return probabilities;
}

double certainty(const std::vector<double>& probabilities) {
    double entropy = 0.0;
    for (const double value : probabilities)
        if (value > 0.0) entropy -= value * std::log(value);
    return probabilities.size() <= 1
        ? 1.0
        : std::max(0.0, 1.0 - entropy / std::log(static_cast<double>(probabilities.size())));
}

double choice_confidence(const std::vector<double>& probabilities) {
    const auto values = normalized(probabilities);
    const double n = static_cast<double>(values.size());
    return values.size() <= 1
        ? 1.0
        : clip01((n * *std::max_element(values.begin(), values.end()) - 1.0) /
                 (n - 1.0));
}

double score_confidence(const std::vector<double>& probabilities) {
    const auto values = normalized(probabilities);
    if (values.size() <= 1) return 1.0;
    const std::size_t peak = static_cast<std::size_t>(
        std::distance(values.begin(),
                      std::max_element(values.begin(), values.end())));
    double spread = 0.0;
    double uniform = 0.0;
    const double middle = (static_cast<double>(values.size()) - 1.0) / 2.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        spread += values[index] * std::abs(static_cast<double>(index) - peak);
        uniform += std::abs(static_cast<double>(index) - middle);
    }
    uniform /= static_cast<double>(values.size());
    return clip01(1.0 - spread / uniform);
}

double rounded(double value, int digits) {
    std::array<char, 128> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::fixed, digits);
    if (converted.ec != std::errc())
        throw std::runtime_error("could not round Decider answer");
    return std::stod(std::string(buffer.data(), converted.ptr));
}

Json format_answer(const Question& question, std::vector<double> probabilities) {
    if (probabilities.size() < question.options.size())
        throw std::invalid_argument("an answer has fewer probabilities than options");
    probabilities.resize(question.options.size());
    probabilities = format_normalized(std::move(probabilities));
    const std::size_t peak = static_cast<std::size_t>(
        std::distance(probabilities.begin(),
                      std::max_element(probabilities.begin(), probabilities.end())));
    Json output = Json::object();
    output["type"] = answer_type_name(question.type);
    if (question.type == AnswerType::Noul) {
        output["noul"] = rounded(probabilities.at(1), 4);
        return output;
    }
    if (question.type == AnswerType::Choice) {
        output["choice"] = question.option_names[peak];
        output["confidence"] = rounded(choice_confidence(probabilities), 4);
    } else {
        double score = 0.0;
        for (std::size_t index = 0; index < probabilities.size(); ++index)
            score += static_cast<double>(index) * probabilities[index];
        output["score"] = rounded(score, 2);
        output["confidence"] = rounded(score_confidence(probabilities), 4);
    }
    output["x_p_max"] = rounded(probabilities[peak], 4);
    output["certainty"] = rounded(certainty(probabilities), 4);
    if (question.type == AnswerType::Score) {
        Json legend = Json::object();
        for (std::size_t index = 0; index < question.legend.size(); ++index)
            legend[std::to_string(index)] = question.legend[index];
        output["legend"] = std::move(legend);
    }
    Json values = Json::object();
    for (std::size_t index = 0; index < probabilities.size(); ++index)
        values[question.option_names[index]] = rounded(probabilities[index], 4);
    output["probabilities"] = std::move(values);
    return output;
}

Question parse_question(const std::string& id, const Json& specification) {
    if (!specification.is_object()) throw std::invalid_argument("question specification must be an object");
    Question question;
    question.id = id;
    const std::string type = specification.value("type", "choice");
    const Json* criteria = optional_member(specification, "criteria");
    if (criteria == nullptr) criteria = optional_member(specification, "options");
    if (criteria != nullptr && criteria->is_null()) criteria = nullptr;
    const Json* raw_instructions = optional_member(specification, "instructions");
    if (raw_instructions == nullptr) raw_instructions = optional_member(specification, "question");
    const bool is_noul = type == "noul" || type == "bool";
    if (is_noul && (raw_instructions == nullptr || missing_description(*raw_instructions))) {
        bool described = false;
        if (criteria != nullptr && criteria->is_object()) {
            for (const char* key : {"true", "false"}) {
                const Json* value = optional_member(*criteria, key);
                described = described || (value != nullptr && !missing_description(*value));
            }
        }
        if (!described && (criteria == nullptr || criteria->is_object()))
            throw std::invalid_argument("noul question without instructions: criteria must describe true or false");
        question.instructions = std::string(kNoulWithoutInstructions);
    } else {
        question.instructions = raw_instructions == nullptr ? std::string() : text(*raw_instructions);
    }
    if (question.instructions.empty()) throw std::invalid_argument("question without instructions");
    const Json* isolated = optional_member(specification, "isolated");
    question.isolated = isolated == nullptr ? true : isolated->get<bool>();

    if (type == "choice") {
        question.type = AnswerType::Choice;
        Json choices;
        if (criteria != nullptr && criteria->is_array()) {
            choices = Json::object();
            for (const auto& value : *criteria) choices[python_str(value)] = nullptr;
            criteria = &choices;
        }
        if (criteria == nullptr || !criteria->is_object() || criteria->size() < 2 ||
            criteria->size() > kMaximumChoiceOptions)
            throw std::invalid_argument("choice criteria: a map of 2..255 options");
        for (const auto& item : criteria->items()) {
            question.option_names.push_back(item.key());
            question.options.push_back(missing_description(item.value())
                ? item.key() : item.key() + ": " + text(item.value()));
        }
    } else if (type == "score") {
        question.type = AnswerType::Score;
        std::vector<Json> levels;
        if (criteria != nullptr && criteria->is_array()) {
            for (const auto& value : *criteria) levels.push_back(value);
        } else if (criteria != nullptr && criteria->is_object()) {
            std::vector<std::pair<double, Json>> sorted;
            for (const auto& item : criteria->items()) {
                std::size_t consumed = 0;
                const double level = std::stod(item.key(), &consumed);
                while (consumed < item.key().size() &&
                       std::isspace(static_cast<unsigned char>(item.key()[consumed])))
                    ++consumed;
                if (consumed != item.key().size())
                    throw std::invalid_argument("score criteria level is not numeric: " + item.key());
                sorted.emplace_back(level, item.value());
            }
            std::stable_sort(sorted.begin(), sorted.end(), [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
            for (auto& item : sorted) levels.push_back(std::move(item.second));
        }
        if (levels.size() < 2 || levels.size() > kMaximumScoreLevels)
            throw std::invalid_argument("score criteria: an ordered list of 2..10 level descriptions");
        for (std::size_t index = 0; index < levels.size(); ++index) {
            question.option_names.push_back(std::to_string(index));
            question.legend.push_back(text(levels[index]));
            question.options.push_back(std::to_string(index) + ": " + text(levels[index]));
        }
    } else if (is_noul) {
        question.type = AnswerType::Noul;
        if (criteria != nullptr && !criteria->is_object())
            throw std::invalid_argument("noul criteria: a map of optional true/false descriptions");
        question.option_names = {"false", "true"};
        const Json empty = Json::object();
        const Json& values = criteria == nullptr ? empty : *criteria;
        const Json* false_value = optional_member(values, "false");
        const Json* true_value = optional_member(values, "true");
        question.options.push_back(false_value == nullptr || missing_description(*false_value)
            ? "no" : "no: " + text(*false_value));
        question.options.push_back(true_value == nullptr || missing_description(*true_value)
            ? "yes" : "yes: " + text(*true_value));
    } else {
        throw std::invalid_argument("unknown question type '" + type + "'");
    }
    return question;
}

}  // namespace

std::string python_json(const Json& value) {
    std::string output;
    append_python_json(output, value);
    return output;
}

Json annotate_indices(const Json& value, std::size_t minimum_length) {
    if (value.is_array()) {
        Json output = Json::array();
        for (std::size_t index = 0; index < value.size(); ++index) {
            Json annotated = annotate_indices(value[index], minimum_length);
            if (value.size() >= minimum_length) {
                Json item = Json::object();
                item["_index"] = index;
                if (annotated.is_object()) {
                    for (const auto& member : annotated.items()) item[member.key()] = member.value();
                } else {
                    item["value"] = std::move(annotated);
                }
                output.push_back(std::move(item));
            } else {
                output.push_back(std::move(annotated));
            }
        }
        return output;
    }
    if (value.is_object()) {
        Json output = Json::object();
        for (const auto& item : value.items())
            output[item.key()] = annotate_indices(item.value(), minimum_length);
        return output;
    }
    return value;
}

RequestPlan parse_system_one(std::string_view request_json, bool isolated_levels) {
    const Json request = Json::parse(request_json.begin(), request_json.end());
    if (!request.is_object()) throw std::invalid_argument("request must be a JSON object");
    const Json* state = optional_member(request, "state");
    const Json* questions = optional_member(request, "questions");
    if (state == nullptr) throw std::invalid_argument("request requires state");
    if (questions == nullptr || !questions->is_object())
        throw std::invalid_argument("request requires a questions object");
    RequestPlan plan;
    plan.independent = request.value("independent", true);
    plan.rendered_state = state->is_string()
        ? state->get<std::string>() : python_json(annotate_indices(*state));
    for (const auto& item : questions->items())
        plan.questions.push_back(parse_question(item.key(), item.value()));
    for (std::size_t index = 0; index < plan.questions.size(); ++index) {
        const Question& question = plan.questions[index];
        if (isolated_levels && plan.independent && question.type == AnswerType::Score && question.isolated) {
            for (std::size_t level = 0; level < question.legend.size(); ++level) {
                Row row;
                row.question_index = index;
                row.temperature_type = AnswerType::Score;
                row.question = question.instructions + std::string(kIsolatedTemplatePrefix) +
                    strip_level_number(question.legend[level]) + std::string(kIsolatedTemplateSuffix);
                row.options = {"no", "yes"};
                row.isolated_level = true;
                row.isolated_level_index = level;
                plan.rows.push_back(std::move(row));
            }
        } else {
            Row row;
            row.question_index = index;
            row.temperature_type = question.type;
            row.question = question.instructions;
            row.options = question.options;
            plan.rows.push_back(std::move(row));
        }
    }
    return plan;
}

std::string plain_narrow_block(const Row& row, std::size_t answer_index, bool multi_question) {
    if (row.options.size() > 10) throw std::invalid_argument("plain_narrow_block received a wide question");
    std::string output = "\n\nQuestion";
    if (multi_question) output += " " + std::to_string(answer_index + 1);
    output += ": " + row.question + "\nOptions:";
    for (std::size_t index = 0; index < row.options.size(); ++index) {
        output += "\n(";
        output.push_back(kLetters[index]);
        output += ") " + row.options[index];
    }
    output += "\nAnswer";
    if (multi_question) output += " " + std::to_string(answer_index + 1);
    output += ": (";
    return output;
}

Json assemble_response(const RequestPlan& plan,
                       const std::vector<std::vector<double>>& row_probabilities,
                       std::string_view model_name,
                       std::size_t input_tokens) {
    if (row_probabilities.size() != plan.rows.size())
        throw std::invalid_argument("probability row count does not match request plan");
    std::vector<std::vector<double>> answers(plan.questions.size());
    std::vector<std::vector<double>> fits(plan.questions.size());
    for (std::size_t row_index = 0; row_index < plan.rows.size(); ++row_index) {
        const Row& row = plan.rows[row_index];
        if (row.isolated_level) {
            if (row_probabilities[row_index].size() < 2)
                throw std::invalid_argument("isolated score row requires no/yes probabilities");
            if (fits[row.question_index].empty())
                fits[row.question_index].resize(plan.questions[row.question_index].legend.size());
            fits[row.question_index][row.isolated_level_index] = row_probabilities[row_index][1];
        } else {
            answers[row.question_index] = row_probabilities[row_index];
        }
    }

    Json answer_object = Json::object();
    for (std::size_t index = 0; index < plan.questions.size(); ++index) {
        if (!fits[index].empty()) {
            const double mass = std::accumulate(fits[index].begin(), fits[index].end(), 0.0);
            answers[index] = fits[index];
            Json answer = format_answer(plan.questions[index], answers[index]);
            Json level_fit = Json::object();
            for (std::size_t level = 0; level < fits[index].size(); ++level)
                level_fit[std::to_string(level)] = rounded(fits[index][level], 4);
            answer["level_fit"] = std::move(level_fit);
            answer["fit_mass"] = rounded(mass, 4);
            answer_object[plan.questions[index].id] = std::move(answer);
        } else {
            answer_object[plan.questions[index].id] =
                format_answer(plan.questions[index], answers[index]);
        }
    }
    Json response = Json::object();
    response["model"] = std::string(model_name);
    response["answers"] = std::move(answer_object);
    response["usage"] = Json{{"input_tokens", input_tokens}, {"output_tokens", 0}};
    return response;
}

const char* answer_type_name(AnswerType type) noexcept {
    switch (type) {
        case AnswerType::Choice: return "choice";
        case AnswerType::Score: return "score";
        case AnswerType::Noul: return "noul";
    }
    return "choice";
}

}  // namespace decider::native
