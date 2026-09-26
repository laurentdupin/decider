#include "decider_protocol.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_python_json_and_annotations() {
    const auto value = decider::native::Json::parse(
        R"({"name":"café","items":[0,1,2,3,4,5,6,{"x":7}]})");
    const auto annotated = decider::native::annotate_indices(value);
    require(decider::native::python_json(annotated) ==
        R"({"name": "café", "items": [{"_index": 0, "value": 0}, {"_index": 1, "value": 1}, {"_index": 2, "value": 2}, {"_index": 3, "value": 3}, {"_index": 4, "value": 4}, {"_index": 5, "value": 5}, {"_index": 6, "value": 6}, {"_index": 7, "x": 7}]})",
        "Python-compatible JSON rendering or array annotation changed");
    const auto numbers = decider::native::Json::parse(
        R"({"negative_zero":-0.0,"small":1e-7,"edge":1e-6,"large":1e20,"large_edge":1e21,"integer":9007199254740993})");
    require(decider::native::python_json(numbers) ==
        R"({"negative_zero": -0.0, "small": 1e-07, "edge": 1e-06, "large": 1e+20, "large_edge": 1e+21, "integer": 9007199254740993})",
        "Python-compatible numeric JSON rendering changed");
}

void test_planning_and_formatting() {
    const std::string request = R"({
      "state":{"ticket":"charged twice"},
      "questions":{
        "department":{"type":"choice","instructions":"Which team?","criteria":{"billing":"charges","support":null}},
        "urgency":{"type":"score","instructions":"How urgent?","criteria":["low","medium","high"]},
        "refund":{"type":"noul","instructions":"Is a refund requested?"}
      },
      "independent":true
    })";
    const auto plan = decider::native::parse_system_one(request, true);
    require(plan.rendered_state == R"({"ticket": "charged twice"})", "state rendering changed");
    require(plan.questions.size() == 3, "question count changed");
    require(plan.rows.size() == 5, "isolated score planning changed");
    require(plan.rows[1].temperature_type == decider::native::AnswerType::Score,
            "isolated score temperature type changed");
    require(plan.rows[1].question ==
        "How urgent?\nProposed answer: low\nDoes the proposed answer fit?",
        "isolated score prompt changed");
    require(decider::native::plain_narrow_block(plan.rows[0], 0, false) ==
        "\n\nQuestion: Which team?\nOptions:\n(A) billing: charges\n(B) support\nAnswer: (",
        "plain narrow prompt block changed");

    const auto response = decider::native::assemble_response(
        plan,
        {{0.8, 0.2}, {0.9, 0.1}, {0.2, 0.8}, {0.8, 0.2}, {0.25, 0.75}},
        "decider-native-test", 42);
    require(response["answers"]["department"]["choice"] == "billing",
            "choice formatting changed");
    require(response["answers"]["urgency"]["score"] == 1.09,
            "isolated score normalization changed");
    require(response["answers"]["urgency"]["fit_mass"] == 1.1,
            "isolated fit mass changed");
    require(response["answers"]["refund"]["noul"] == 0.75,
            "noul formatting changed");
}

void test_isolation_requires_independence() {
    const auto plan = decider::native::parse_system_one(
        R"({"state":"x","questions":{"s":{"type":"score","instructions":"S?","criteria":["a","b"]}},"independent":false})",
        true);
    require(plan.rows.size() == 1 && !plan.rows[0].isolated_level,
            "packed requests must not isolate score levels");
}

void test_packed_prompt_numbering() {
    const auto plan = decider::native::parse_system_one(
        R"({"state":"x","questions":{"a":{"type":"choice","instructions":"First?","criteria":["x","y"]},"b":{"type":"noul","instructions":"Second?"}},"independent":false})",
        true);
    require(plan.rows.size() == 2 && !plan.independent,
            "packed planning changed");
    require(decider::native::plain_narrow_block(plan.rows[0], 0, true) ==
        "\n\nQuestion 1: First?\nOptions:\n(A) x\n(B) y\nAnswer 1: (",
        "first packed prompt block changed");
    require(decider::native::plain_narrow_block(plan.rows[1], 1, true) ==
        "\n\nQuestion 2: Second?\nOptions:\n(A) no\n(B) yes\nAnswer 2: (",
        "second packed prompt block changed");
}

void test_python_protocol_edge_cases() {
    const auto noul = decider::native::parse_system_one(
        R"({"state":"x","questions":{"n":{"type":"noul","instructions":"N?","criteria":null}}})",
        false);
    require(noul.rows[0].options == std::vector<std::string>({"no", "yes"}),
            "explicit null Noul criteria changed");
    require_throws([] {
        decider::native::parse_system_one(
            R"({"state":"x","questions":{"n":{"type":"noul","criteria":null}}})", false);
    }, "description-free Noul with null criteria must fail");

    const auto choice = decider::native::parse_system_one(
        R"({"state":"x","questions":{"c":{"type":"choice","instructions":"C?","criteria":[true,null]}}})",
        false);
    require(choice.questions[0].option_names ==
                std::vector<std::string>({"True", "None"}),
            "choice list must use Python str() names");
    require_throws([] {
        decider::native::parse_system_one(
            R"({"state":"x","questions":{"s":{"type":"score","instructions":"S?","criteria":{"1junk":"bad","2":"ok"}}}})",
            false);
    }, "score legend numeric prefixes must fail");

    const auto isolated = decider::native::parse_system_one(
        R"({"state":"x","questions":{"s":{"type":"score","instructions":"S?","criteria":["a","b"]}}})",
        true);
    const auto zero = decider::native::assemble_response(
        isolated, {{1.0, 0.0}, {1.0, 0.0}}, "m", 1);
    require(zero["answers"]["s"]["score"] == 0.0 &&
                zero["answers"]["s"]["x_p_max"] == 0.0 &&
                zero["answers"]["s"]["certainty"] == 1.0,
            "all-zero isolated fits must remain zero");

    const auto rounded = decider::native::assemble_response(
        noul, {{0.99995, 0.00005}}, "m", 1);
    require(rounded["answers"]["n"]["noul"] == 0.0001,
            "answer rounding must match Python decimal rounding");
}

}  // namespace

int main() {
    try {
        test_python_json_and_annotations();
        test_planning_and_formatting();
        test_isolation_requires_independence();
        test_packed_prompt_numbering();
        test_python_protocol_edge_cases();
        std::cout << "DECIDER_NATIVE_PROTOCOL_OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
