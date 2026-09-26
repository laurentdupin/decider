#include "decider_protocol.h"

#include <inferbridge/inferbridge_harness.h>
#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

constexpr char kHarnessId[] = "inferbridge.decider.gguf";
constexpr char kHarnessVersion[] = "0.1.0";
constexpr std::uint64_t kDefaultOutputBytes = 1024U * 1024U;
constexpr std::uint64_t kMaximumOutputBytes = 16U * 1024U * 1024U;

std::once_flag g_backend_once;
std::mutex g_error_mutex;
std::string g_last_error;

std::string as_string(ibrh_string_view value) {
    return value.data == nullptr ? std::string() : std::string(value.data, value.size);
}

ibrh_string_view view(const std::string& value) {
    return {value.data(), value.size()};
}

void set_error(std::string message) {
    std::lock_guard<std::mutex> guard(g_error_mutex);
    g_last_error = std::move(message);
}

bool valid_api(std::uint32_t version) {
    return (version >> 16U) == IBRH_API_VERSION_MAJOR;
}

template <typename T>
bool valid_struct(std::size_t size, const T* value) {
    return value != nullptr && size >= sizeof(T) && value->struct_size >= sizeof(T) &&
        valid_api(value->api_version);
}

void quiet_llama_log(ggml_log_level, const char*, void*) {}

decider::native::Json read_json_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::invalid_argument("decider_config.json was not found beside the GGUF model");
    return decider::native::Json::parse(input, nullptr, true, true);
}

bool has_vulkan_gpu() {
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        std::string name = registry == nullptr ? std::string() : ggml_backend_reg_name(registry);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            name.find("vulkan") != std::string::npos)
            return true;
    }
    return false;
}

struct ModelConfig {
    std::uint32_t context_size = 32768;
    std::uint32_t batch_size = 512;
    std::uint32_t threads = std::max(1U, std::thread::hardware_concurrency());
    std::uint32_t max_rows = 256;
    std::uint32_t max_outputs_per_batch = 32;
    std::uint32_t max_state_tokens = 32768;
    std::uint64_t max_output_bytes = kDefaultOutputBytes;
    std::string model_name = "decider-native";
    bool isolated_levels = false;
    double temperature = 1.0;
    double choice_temperature = 1.0;
    double score_temperature = 1.0;
    double noul_temperature = 1.0;
};

void require_temperature(double value, const char* label) {
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string(label) + " must be finite and positive");
}

double base_temperature(const decider::native::Json& value, const char* label) {
    if (value.is_boolean())
        throw std::invalid_argument(std::string(label) + " must be a finite number > 0");
    double result = 0.0;
    if (value.is_number()) {
        result = value.get<double>();
    } else if (value.is_string()) {
        std::size_t consumed = 0;
        result = std::stod(value.get_ref<const std::string&>(), &consumed);
        const auto& source = value.get_ref<const std::string&>();
        while (consumed < source.size() &&
               std::isspace(static_cast<unsigned char>(source[consumed])))
            ++consumed;
        if (consumed != source.size())
            throw std::invalid_argument(std::string(label) + " must be a finite number > 0");
    } else {
        throw std::invalid_argument(std::string(label) + " must be a finite number > 0");
    }
    require_temperature(result, label);
    return result;
}

std::string python_scalar_string(const decider::native::Json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_null()) return "None";
    if (value.is_boolean()) return value.get<bool>() ? "True" : "False";
    return value.dump();
}

ModelConfig parse_model_config(const decider::native::Json& checkpoint,
                               const decider::native::Json& parameters) {
    ModelConfig result;
    result.context_size = parameters.value("context_size", result.context_size);
    result.batch_size = parameters.value("batch_size", result.batch_size);
    result.threads = parameters.value("threads", result.threads);
    result.max_rows = parameters.value("max_rows", result.max_rows);
    result.max_outputs_per_batch = parameters.value(
        "max_outputs_per_batch", result.max_outputs_per_batch);
    result.max_state_tokens = parameters.value("max_state_tokens", result.max_state_tokens);
    result.max_output_bytes = parameters.value("max_output_bytes", result.max_output_bytes);
    result.isolated_levels = checkpoint.value("isolated_levels", false);
    if (const auto found = checkpoint.find("temperature"); found != checkpoint.end())
        result.temperature = base_temperature(*found, "temperature");
    result.choice_temperature = result.temperature;
    result.score_temperature = result.temperature;
    result.noul_temperature = result.temperature;
    if (const auto found = checkpoint.find("temperature_by_type"); found != checkpoint.end() && !found->is_null()) {
        if (!found->is_object())
            throw std::invalid_argument("temperature_by_type must be a map");
        for (const auto& item : found->items()) {
            if (item.key() != "choice" && item.key() != "score" && item.key() != "noul")
                throw std::invalid_argument("temperature_by_type has unknown key: " + item.key());
            if (!item.value().is_number() || item.value().is_boolean())
                throw std::invalid_argument("temperature_by_type values must be finite numbers > 0");
            const double value = item.value().get<double>();
            require_temperature(value, "temperature_by_type value");
            if (item.key() == "choice") result.choice_temperature = value;
            if (item.key() == "score") result.score_temperature = value;
            if (item.key() == "noul") result.noul_temperature = value;
        }
    }
    if (const auto found = parameters.find("temperature"); found != parameters.end()) {
        result.temperature = base_temperature(*found, "temperature");
        result.choice_temperature = result.temperature;
        result.score_temperature = result.temperature;
        result.noul_temperature = result.temperature;
    }
    if (const auto found = parameters.find("isolated_levels"); found != parameters.end())
        result.isolated_levels = found->get<bool>();
    const auto version_member = checkpoint.find("version");
    const std::string version = version_member == checkpoint.end()
        ? "native" : python_scalar_string(*version_member);
    result.model_name = parameters.value("model_name", "decider-" + version);
    const auto chat_member = checkpoint.find("chat_template");
    const bool chat_template = chat_member != checkpoint.end() &&
        chat_member->is_boolean() && chat_member->get<bool>();
    const std::string layout = checkpoint.value("layout", chat_template ? "chat" : "plain");
    if (layout == "plain" && chat_template)
        throw std::invalid_argument(
            "decider_config.json contradicts layout='plain' with chat_template=true");
    if (layout != "plain")
        throw std::invalid_argument(
            "the initial native harness supports only a state-first plain-layout checkpoint");
    if (result.context_size < 128 || result.batch_size == 0 || result.threads == 0 ||
        result.max_rows == 0 || result.max_outputs_per_batch == 0 ||
        result.max_outputs_per_batch > result.batch_size || result.max_state_tokens == 0 ||
        result.max_output_bytes < 1024 || result.max_output_bytes > kMaximumOutputBytes)
        throw std::invalid_argument("invalid Decider native model parameters");
    require_temperature(result.choice_temperature, "choice temperature");
    require_temperature(result.score_temperature, "score temperature");
    require_temperature(result.noul_temperature, "noul temperature");
    return result;
}

std::vector<llama_token> tokenize(const llama_vocab* vocabulary, const std::string& text) {
    int count = llama_tokenize(vocabulary, text.data(), static_cast<int>(text.size()),
                               nullptr, 0, false, true);
    if (count == std::numeric_limits<int>::min())
        throw std::runtime_error("text is too large to tokenize");
    if (count >= 0) return {};
    std::vector<llama_token> tokens(static_cast<std::size_t>(-count));
    count = llama_tokenize(vocabulary, text.data(), static_cast<int>(text.size()),
                           tokens.data(), static_cast<int>(tokens.size()), false, true);
    if (count < 0) throw std::runtime_error("llama.cpp tokenization failed");
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

std::vector<std::string> candidate_label_names() {
    std::vector<std::string> names;
    names.reserve(26 + 26 * 26);
    for (char first = 'A'; first <= 'Z'; ++first)
        names.emplace_back(1, first);
    for (char first = 'A'; first <= 'Z'; ++first)
        for (char second = 'A'; second <= 'Z'; ++second)
            names.push_back(std::string(1, first) + second);
    return names;
}

void fill_text_port(std::uint32_t direction, std::uint32_t width,
                    ibrh_port_descriptor* descriptor) {
    *descriptor = {};
    descriptor->struct_size = sizeof(*descriptor);
    descriptor->api_version = IBRH_CURRENT_API_VERSION;
    descriptor->index = 0;
    descriptor->direction = direction;
    descriptor->semantic = IBRH_SEMANTIC_TEXT;
    descriptor->payload_type = IBRH_PAYLOAD_UTF8_JSON;
    descriptor->pixel_format = IBRH_PAYLOAD_UTF8_JSON;
    descriptor->resource_kind = IBRH_RESOURCE_KIND_BUFFER;
    descriptor->width = width;
    descriptor->height = 1;
    descriptor->depth = 1;
    descriptor->flags = width == 0 ? IBRH_DESCRIPTOR_DYNAMIC_WIDTH : 0;
    descriptor->accepted_pixel_format_mask = 1ULL << IBRH_PAYLOAD_UTF8_JSON;
}

void validate_host_json_binding(const ibrh_transfer_binding& binding,
                                std::uint32_t access) {
    const auto& resource = binding.resource;
    if (binding.struct_size < sizeof(binding) ||
        binding.api_version != IBRH_CURRENT_API_VERSION ||
        resource.struct_size < sizeof(resource) ||
        resource.api_version != IBRH_CURRENT_API_VERSION ||
        resource.domain != IBRH_RESOURCE_DOMAIN_HOST ||
        resource.kind != IBRH_RESOURCE_KIND_BUFFER ||
        resource.access != access ||
        resource.pixel_format != IBRH_PAYLOAD_UTF8_JSON ||
        resource.native_handle_type != IBRH_NATIVE_HANDLE_HOST_POINTER ||
        resource.native_handle == 0 || resource.byte_size == 0 ||
        resource.byte_offset > std::numeric_limits<std::uint64_t>::max() - resource.byte_size)
        throw std::invalid_argument("expected one host UTF-8 JSON buffer binding");
}

double temperature_for(const ModelConfig& config, decider::native::AnswerType type) {
    switch (type) {
        case decider::native::AnswerType::Choice: return config.choice_temperature;
        case decider::native::AnswerType::Score: return config.score_temperature;
        case decider::native::AnswerType::Noul: return config.noul_temperature;
    }
    return config.temperature;
}

std::vector<double> selected_softmax(const float* logits,
                                     const std::vector<llama_token>& labels,
                                     std::size_t count, double temperature,
                                     std::vector<double>* raw_logits = nullptr) {
    if (count < 2 || count > labels.size())
        throw std::invalid_argument("the native harness supports 2..255 options per row");
    std::vector<double> values(count);
    if (raw_logits != nullptr) {
        raw_logits->clear();
        raw_logits->reserve(count);
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < count; ++index) {
        const double raw = static_cast<double>(logits[labels[index]]);
        if (raw_logits != nullptr) raw_logits->push_back(raw);
        values[index] = raw / temperature;
        maximum = std::max(maximum, values[index]);
    }
    double total = 0.0;
    for (double& value : values) {
        value = std::exp(value - maximum);
        total += value;
    }
    for (double& value : values) value /= total;
    return values;
}

std::size_t unique_tokens(const std::vector<std::vector<llama_token>>& rows) {
    if (rows.empty()) return 0;
    if (rows.size() == 1) return rows.front().size();
    std::size_t shortest = rows.front().size();
    for (const auto& row : rows) shortest = std::min(shortest, row.size());
    std::size_t prefix = 0;
    while (prefix < shortest) {
        const llama_token token = rows.front()[prefix];
        if (!std::all_of(rows.begin() + 1, rows.end(),
                         [&](const auto& row) { return row[prefix] == token; })) break;
        ++prefix;
    }
    std::size_t total = prefix;
    for (const auto& row : rows) total += row.size() - prefix;
    return total;
}

}  // namespace

struct ibrh_runtime {
    std::string backend;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    const llama_vocab* vocabulary = nullptr;
    ModelConfig config;
    std::vector<llama_token> labels;
    std::atomic<bool> active{false};
};

struct ibrh_job {
    ibrh_model* model = nullptr;
    std::uint64_t source_frame_id = 0;
    std::atomic<std::uint32_t> state{IBRH_JOB_QUEUED};
    std::atomic<bool> cancel{false};
    bool diagnostics = false;
    std::thread worker;
    std::string error;
};

namespace {

struct PreparedRow {
    std::vector<llama_token> tokens;
    std::vector<std::size_t> slots;
    std::vector<std::size_t> option_counts;
    std::vector<decider::native::AnswerType> types;
};

struct EvaluatedRow {
    std::vector<std::vector<double>> probabilities;
    std::vector<std::vector<double>> selected_logits;
};

void append_prompt_block(const ibrh_model* model, PreparedRow& prepared,
                         const decider::native::Row& row,
                         std::size_t answer_index, bool multi_question) {
    if (row.options.size() <= 10) {
        auto block = tokenize(model->vocabulary,
            decider::native::plain_narrow_block(row, answer_index, multi_question));
        prepared.tokens.insert(prepared.tokens.end(), block.begin(), block.end());
        return;
    }
    std::string header = "\n\nQuestion";
    if (multi_question) header += " " + std::to_string(answer_index + 1);
    header += ": " + row.question + "\nOptions:";
    auto piece = tokenize(model->vocabulary, header);
    prepared.tokens.insert(prepared.tokens.end(), piece.begin(), piece.end());
    const auto open = tokenize(model->vocabulary, "\n(");
    for (std::size_t index = 0; index < row.options.size(); ++index) {
        prepared.tokens.insert(prepared.tokens.end(), open.begin(), open.end());
        prepared.tokens.push_back(model->labels[index]);
        piece = tokenize(model->vocabulary, ") " + row.options[index]);
        prepared.tokens.insert(prepared.tokens.end(), piece.begin(), piece.end());
    }
    std::string tail = "\nAnswer";
    if (multi_question) tail += " " + std::to_string(answer_index + 1);
    tail += ": (";
    piece = tokenize(model->vocabulary, tail);
    prepared.tokens.insert(prepared.tokens.end(), piece.begin(), piece.end());
}

EvaluatedRow evaluate_row(ibrh_job* job, const PreparedRow& row) {
    ibrh_model* model = job->model;
    if (row.tokens.empty()) throw std::runtime_error("Decider prompt produced no tokens");
    if (row.tokens.size() > model->config.context_size)
        throw std::runtime_error("Decider prompt exceeds context_size");
    if (row.slots.size() != row.option_counts.size() || row.slots.size() != row.types.size())
        throw std::logic_error("Decider prompt slot metadata is inconsistent");
    llama_memory_clear(llama_get_memory(model->context), true);
    EvaluatedRow evaluated;
    evaluated.probabilities.reserve(row.slots.size());
    evaluated.selected_logits.reserve(row.slots.size());
    std::size_t offset = 0;
    std::size_t next_slot = 0;
    while (offset < row.tokens.size()) {
        if (job->cancel.load()) throw std::runtime_error("cancelled");
        std::size_t end = std::min<std::size_t>(
            offset + model->config.batch_size, row.tokens.size());
        const auto first = std::lower_bound(row.slots.begin(), row.slots.end(), offset);
        const auto limit = first + std::min<std::size_t>(
            model->config.max_outputs_per_batch,
            static_cast<std::size_t>(row.slots.end() - first));
        if (limit != row.slots.end() && *limit < end) end = *limit;
        const std::size_t count = end - offset;
        llama_batch batch = llama_batch_init(static_cast<int>(count), 0, 1);
        batch.n_tokens = static_cast<int>(count);
        const std::size_t chunk_slot_begin = next_slot;
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t global = offset + local;
            batch.token[local] = row.tokens[global];
            batch.pos[local] = static_cast<llama_pos>(global);
            batch.n_seq_id[local] = 1;
            batch.seq_id[local][0] = 0;
            batch.logits[local] = next_slot < row.slots.size() && row.slots[next_slot] == global;
            if (batch.logits[local]) ++next_slot;
        }
        const int result = llama_decode(model->context, batch);
        if (result != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama.cpp prompt decode failed with code " +
                                     std::to_string(result));
        }
        for (std::size_t slot = chunk_slot_begin; slot < next_slot; ++slot) {
            const int local = static_cast<int>(row.slots[slot] - offset);
            const float* logits = llama_get_logits_ith(model->context, local);
            if (logits == nullptr) {
                llama_batch_free(batch);
                throw std::runtime_error("llama.cpp returned no answer-slot logits");
            }
            std::vector<double> raw;
            evaluated.probabilities.push_back(selected_softmax(
                logits, model->labels, row.option_counts[slot],
                temperature_for(model->config, row.types[slot]),
                job->diagnostics ? &raw : nullptr));
            if (job->diagnostics) evaluated.selected_logits.push_back(std::move(raw));
        }
        llama_batch_free(batch);
        offset += count;
    }
    if (evaluated.probabilities.size() != row.slots.size())
        throw std::runtime_error("not all Decider answer slots produced logits");
    return evaluated;
}

void run_job(ibrh_job* job, std::string request_text,
             char* output, std::size_t output_capacity) noexcept {
    try {
        job->state = IBRH_JOB_RUNNING;
        auto plan = decider::native::parse_system_one(
            request_text, job->model->config.isolated_levels);
        if (plan.rows.size() > job->model->config.max_rows)
            throw std::invalid_argument("request expands beyond max_rows");

        std::vector<PreparedRow> prepared_rows;
        prepared_rows.reserve(plan.independent ? plan.rows.size() : 1);
        const auto state_tokens = [&] {
            auto tokens = tokenize(job->model->vocabulary,
                "Context:\n" + plan.rendered_state);
            if (tokens.size() > job->model->config.max_state_tokens)
                tokens.resize(job->model->config.max_state_tokens);
            return tokens;
        }();
        for (std::size_t row_index = 0; row_index < plan.rows.size(); ++row_index) {
            const auto& row = plan.rows[row_index];
            if (plan.independent || prepared_rows.empty()) {
                prepared_rows.push_back(PreparedRow{});
                prepared_rows.back().tokens = state_tokens;
            }
            PreparedRow& prepared = prepared_rows.back();
            append_prompt_block(job->model, prepared, row,
                plan.independent ? 0 : row_index,
                !plan.independent && plan.rows.size() > 1);
            prepared.slots.push_back(prepared.tokens.size() - 1);
            prepared.option_counts.push_back(row.options.size());
            prepared.types.push_back(row.temperature_type);
        }

        std::vector<std::vector<double>> probabilities;
        probabilities.reserve(plan.rows.size());
        decider::native::Json diagnostic_rows = decider::native::Json::array();
        for (const auto& prepared : prepared_rows) {
            if (job->cancel.load()) {
                job->state = IBRH_JOB_CANCELLED;
                return;
            }
            auto evaluated = evaluate_row(job, prepared);
            if (job->diagnostics) {
                decider::native::Json item = decider::native::Json::object();
                item["token_ids"] = prepared.tokens;
                item["slots"] = prepared.slots;
                item["option_counts"] = prepared.option_counts;
                decider::native::Json types = decider::native::Json::array();
                for (const auto type : prepared.types)
                    types.push_back(decider::native::answer_type_name(type));
                item["types"] = std::move(types);
                item["selected_logits"] = std::move(evaluated.selected_logits);
                item["probabilities"] = evaluated.probabilities;
                diagnostic_rows.push_back(std::move(item));
            }
            probabilities.insert(probabilities.end(),
                                 std::make_move_iterator(evaluated.probabilities.begin()),
                                 std::make_move_iterator(evaluated.probabilities.end()));
        }
        std::vector<std::vector<llama_token>> token_rows;
        token_rows.reserve(prepared_rows.size());
        for (const auto& row : prepared_rows) token_rows.push_back(row.tokens);
        auto response = decider::native::assemble_response(
            plan, probabilities, job->model->config.model_name,
            unique_tokens(token_rows));
        if (job->diagnostics) {
            decider::native::Json diagnostics = decider::native::Json::object();
            diagnostics["label_token_ids"] = job->model->labels;
            diagnostics["rows"] = std::move(diagnostic_rows);
            response["_diagnostics"] = std::move(diagnostics);
        }
        const std::string serialized = response.dump();
        if (serialized.size() + 1 > output_capacity)
            throw std::runtime_error("response exceeds configured max_output_bytes");
        std::memset(output, 0, output_capacity);
        std::memcpy(output, serialized.data(), serialized.size());
        job->state = IBRH_JOB_COMPLETE;
    } catch (const std::exception& exception) {
        if (job->cancel.load() || std::string(exception.what()) == "cancelled") {
            job->state = IBRH_JOB_CANCELLED;
        } else {
            job->error = exception.what();
            set_error(job->error);
            job->state = IBRH_JOB_FAILED;
        }
    } catch (...) {
        job->error = "unknown native Decider failure";
        set_error(job->error);
        job->state = IBRH_JOB_FAILED;
    }
}

ibrh_result IBRH_CALL query_capabilities(std::size_t size,
                                         ibrh_capabilities* output) {
    if (output == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *output = {};
    output->struct_size = sizeof(*output);
    output->api_version = IBRH_CURRENT_API_VERSION;
    output->flags = IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_HOST_MEMORY;
    output->input_domain_mask = 1ULL << IBRH_RESOURCE_DOMAIN_HOST;
    output->output_domain_mask = 1ULL << IBRH_RESOURCE_DOMAIN_HOST;
    output->maximum_inputs = 1;
    output->maximum_outputs = 1;
    output->maximum_in_flight_jobs = 1;
    output->harness_id = {kHarnessId, sizeof(kHarnessId) - 1};
    output->harness_version = {kHarnessVersion, sizeof(kHarnessVersion) - 1};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(std::size_t size,
                                     const ibrh_runtime_create_request* request,
                                     ibrh_runtime** output) {
    if (!valid_struct(size, request) || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    try {
        std::call_once(g_backend_once, [] {
            llama_log_set(quiet_llama_log, nullptr);
            llama_backend_init();
        });
        auto runtime = std::make_unique<ibrh_runtime>();
        runtime->backend = as_string(request->backend);
        std::transform(runtime->backend.begin(), runtime->backend.end(),
                       runtime->backend.begin(), [](unsigned char value) {
                           return static_cast<char>(std::toupper(value));
                       });
        if (runtime->backend.empty()) runtime->backend = "CPU";
        if (runtime->backend != "CPU" && runtime->backend != "VULKAN")
            throw std::invalid_argument("Decider harness supports CPU or VULKAN backends");
        if (runtime->backend == "VULKAN" && !has_vulkan_gpu())
            throw std::invalid_argument("no llama.cpp Vulkan GPU device is registered");
        *output = runtime.release();
        return IBRH_OK;
    } catch (const std::exception& exception) {
        set_error(exception.what());
        return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
    }
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) { delete runtime; }

ibrh_result IBRH_CALL model_load(ibrh_runtime* runtime, std::size_t size,
                                 const ibrh_model_load_request* request,
                                 ibrh_model** output) {
    if (runtime == nullptr || !valid_struct(size, request) || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    try {
        const auto parameters = as_string(request->parameters_json).empty()
            ? decider::native::Json::object()
            : decider::native::Json::parse(as_string(request->parameters_json));
        std::filesystem::path model_path =
            std::filesystem::u8path(as_string(request->model_path));
        std::filesystem::path model_root = model_path.parent_path();
        if (std::filesystem::is_directory(model_path)) {
            model_root = model_path;
            const std::string filename = parameters.value("model_filename", std::string());
            if (filename.empty())
                throw std::invalid_argument("model directory requires model_filename");
            model_path /= std::filesystem::u8path(filename);
        }
        std::filesystem::path config_path = model_root / "decider_config.json";
        if (const auto found = parameters.find("config_path");
            found != parameters.end() && found->is_string())
            config_path = std::filesystem::u8path(found->get<std::string>());
        const auto checkpoint = read_json_file(config_path);
        auto model = std::make_unique<ibrh_model>();
        model->runtime = runtime;
        model->config = parse_model_config(checkpoint, parameters);

        llama_model_params model_parameters = llama_model_default_params();
        model_parameters.n_gpu_layers = runtime->backend == "VULKAN" ? -1 : 0;
        model_parameters.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_parameters.load_mtp = false;
        model->model = llama_model_load_from_file(model_path.u8string().c_str(), model_parameters);
        if (model->model == nullptr) throw std::runtime_error("llama.cpp failed to load Decider GGUF");
        model->vocabulary = llama_model_get_vocab(model->model);
        for (const auto& label : candidate_label_names()) {
            const auto encoded = tokenize(model->vocabulary, label);
            if (encoded.size() == 1) model->labels.push_back(encoded.front());
            if (model->labels.size() == 255) break;
        }
        if (model->labels.size() != 255 ||
            std::unordered_set<llama_token>(model->labels.begin(), model->labels.end()).size() !=
                model->labels.size()) {
                llama_model_free(model->model);
                model->model = nullptr;
                throw std::runtime_error(
                    "GGUF tokenizer does not provide 255 distinct Decider label tokens");
        }
        llama_context_params context = llama_context_default_params();
        context.n_ctx = model->config.context_size;
        context.n_batch = model->config.batch_size;
        context.n_ubatch = model->config.batch_size;
        context.n_seq_max = 1;
        context.n_outputs_max = model->config.max_outputs_per_batch;
        context.n_outputs_max_per_seq = model->config.max_outputs_per_batch;
        context.n_threads = static_cast<int>(model->config.threads);
        context.n_threads_batch = static_cast<int>(model->config.threads);
        context.no_perf = true;
        model->context = llama_init_from_model(model->model, context);
        if (model->context == nullptr) {
            llama_model_free(model->model);
            model->model = nullptr;
            throw std::runtime_error("llama.cpp failed to create Decider context");
        }
        *output = model.release();
        return IBRH_OK;
    } catch (const std::exception& exception) {
        set_error(exception.what());
        return IBRH_ERROR_INTERNAL;
    }
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
    llama_free(model->context);
    llama_model_free(model->model);
    delete model;
}

ibrh_result IBRH_CALL model_describe_io(const ibrh_model* model, std::size_t size,
                                        ibrh_model_io_descriptor* output) {
    if (model == nullptr || output == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *output = {};
    output->struct_size = sizeof(*output);
    output->api_version = IBRH_CURRENT_API_VERSION;
    output->input_count = 1;
    output->output_count = 1;
    return IBRH_OK;
}

ibrh_result IBRH_CALL model_get_port(const ibrh_model* model,
                                     std::uint32_t direction, std::uint32_t index,
                                     std::size_t size,
                                     ibrh_port_descriptor* output) {
    if (model == nullptr || output == nullptr || index != 0 ||
        (direction != IBRH_PORT_INPUT && direction != IBRH_PORT_OUTPUT))
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    fill_text_port(direction, direction == IBRH_PORT_OUTPUT
        ? static_cast<std::uint32_t>(model->config.max_output_bytes) : 0, output);
    return IBRH_OK;
}

ibrh_result IBRH_CALL model_plan_outputs(const ibrh_model* model, std::size_t size,
                                         const ibrh_output_plan_request* request,
                                         std::uint32_t capacity,
                                         ibrh_port_descriptor* outputs) {
    if (model == nullptr || !valid_struct(size, request) ||
        request->input_count != 1 || request->inputs == nullptr ||
        capacity < 1 || outputs == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    fill_text_port(IBRH_PORT_OUTPUT,
        static_cast<std::uint32_t>(model->config.max_output_bytes), outputs);
    return IBRH_OK;
}

ibrh_result IBRH_CALL submit(ibrh_model* model, std::size_t size,
                             const ibrh_submit_request* request,
                             ibrh_job** output) {
    if (model == nullptr || !valid_struct(size, request) || output == nullptr ||
        request->input_count != 1 || request->output_count != 1 ||
        request->inputs == nullptr || request->outputs == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    bool expected = false;
    if (!model->active.compare_exchange_strong(expected, true))
        return IBRH_ERROR_INVALID_STATE;
    try {
        validate_host_json_binding(request->inputs[0], IBRH_RESOURCE_ACCESS_READ);
        validate_host_json_binding(request->outputs[0], IBRH_RESOURCE_ACCESS_WRITE);
        const auto& input = request->inputs[0].resource;
        const auto& destination = request->outputs[0].resource;
        const auto* input_data = reinterpret_cast<const char*>(
            static_cast<std::uintptr_t>(input.native_handle + input.byte_offset));
        auto* output_data = reinterpret_cast<char*>(
            static_cast<std::uintptr_t>(destination.native_handle + destination.byte_offset));
        auto job = std::make_unique<ibrh_job>();
        job->model = model;
        job->source_frame_id = request->source_frame_id;
        const std::string submit_parameters = as_string(request->parameters_json);
        if (!submit_parameters.empty()) {
            const auto parameters = decider::native::Json::parse(submit_parameters);
            if (!parameters.is_object())
                throw std::invalid_argument("submit parameters_json must be an object");
            job->diagnostics = parameters.value("diagnostics", false);
        }
        std::string request_text(input_data, static_cast<std::size_t>(input.byte_size));
        while (!request_text.empty() && request_text.back() == '\0') request_text.pop_back();
        job->worker = std::thread(run_job, job.get(), std::move(request_text),
            output_data, static_cast<std::size_t>(destination.byte_size));
        *output = job.release();
        return IBRH_OK;
    } catch (const std::exception& exception) {
        model->active = false;
        set_error(exception.what());
        return IBRH_ERROR_INVALID_ARGUMENT;
    }
}

ibrh_result IBRH_CALL job_poll(const ibrh_job* job, std::size_t size,
                               ibrh_job_status* output) {
    if (job == nullptr || output == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *output = {};
    output->struct_size = sizeof(*output);
    output->state = job->state.load();
    output->output_count = output->state == IBRH_JOB_COMPLETE ? 1 : 0;
    output->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (job == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    job->cancel = true;
    return IBRH_OK;
}

void IBRH_CALL job_release(ibrh_job* job) {
    if (job == nullptr) return;
    job->cancel = true;
    if (job->worker.joinable()) job->worker.join();
    job->model->active = false;
    delete job;
}

ibrh_result IBRH_CALL get_last_error(const void*, char* destination,
                                     std::size_t destination_size,
                                     std::size_t* required_size) {
    std::lock_guard<std::mutex> guard(g_error_mutex);
    const std::size_t required = g_last_error.size() + 1;
    if (required_size != nullptr) *required_size = required;
    if (destination == nullptr) return IBRH_OK;
    if (destination_size < required) return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, g_last_error.c_str(), required);
    return IBRH_OK;
}

}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    std::uint32_t requested_version, std::size_t size, ibrh_api* output) {
    if (!valid_api(requested_version)) return IBRH_ERROR_UNSUPPORTED_API;
    if (output == nullptr || size < sizeof(*output)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *output = {};
    output->struct_size = sizeof(*output);
    output->api_version = IBRH_CURRENT_API_VERSION;
    output->query_capabilities = query_capabilities;
    output->runtime_create = runtime_create;
    output->runtime_destroy = runtime_destroy;
    output->model_load = model_load;
    output->model_unload = model_unload;
    output->model_describe_io = model_describe_io;
    output->model_get_port = model_get_port;
    output->model_plan_outputs = model_plan_outputs;
    output->submit = submit;
    output->job_poll = job_poll;
    output->job_cancel = job_cancel;
    output->job_release = job_release;
    output->get_last_error = get_last_error;
    return IBRH_OK;
}
