#include "decider_protocol.h"

#include <inferbridge/inferbridge_harness.h>
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
    if (!input) return decider::native::Json::object();
    return decider::native::Json::parse(input, nullptr, true, true);
}

struct ModelConfig {
    std::uint32_t context_size = 32768;
    std::uint32_t batch_size = 512;
    std::uint32_t threads = std::max(1U, std::thread::hardware_concurrency());
    std::uint32_t max_rows = 256;
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

ModelConfig parse_model_config(const decider::native::Json& checkpoint,
                               const decider::native::Json& parameters) {
    ModelConfig result;
    result.context_size = parameters.value("context_size", result.context_size);
    result.batch_size = parameters.value("batch_size", result.batch_size);
    result.threads = parameters.value("threads", result.threads);
    result.max_rows = parameters.value("max_rows", result.max_rows);
    result.max_state_tokens = parameters.value("max_state_tokens", result.max_state_tokens);
    result.max_output_bytes = parameters.value("max_output_bytes", result.max_output_bytes);
    result.isolated_levels = checkpoint.value("isolated_levels", false);
    result.temperature = checkpoint.value("temperature", 1.0);
    result.choice_temperature = result.temperature;
    result.score_temperature = result.temperature;
    result.noul_temperature = result.temperature;
    if (const auto found = checkpoint.find("temperature_by_type");
        found != checkpoint.end() && found->is_object()) {
        result.choice_temperature = found->value("choice", result.temperature);
        result.score_temperature = found->value("score", result.temperature);
        result.noul_temperature = found->value("noul", result.temperature);
    }
    if (const auto found = parameters.find("temperature"); found != parameters.end()) {
        result.temperature = found->get<double>();
        result.choice_temperature = result.temperature;
        result.score_temperature = result.temperature;
        result.noul_temperature = result.temperature;
    }
    if (const auto found = parameters.find("isolated_levels"); found != parameters.end())
        result.isolated_levels = found->get<bool>();
    const std::string version = checkpoint.value("version", std::string("native"));
    result.model_name = parameters.value("model_name", "decider-" + version);
    const std::string layout = checkpoint.value("layout",
        checkpoint.value("chat_template", false) ? "chat" : "plain");
    if (layout != "plain")
        throw std::invalid_argument(
            "the initial native harness supports only a state-first plain-layout checkpoint");
    if (result.context_size < 128 || result.batch_size == 0 || result.threads == 0 ||
        result.max_rows == 0 || result.max_state_tokens == 0 ||
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
                                     std::size_t count, double temperature) {
    if (count < 2 || count > labels.size())
        throw std::invalid_argument("the initial native harness supports 2..10 options per row");
    std::vector<double> values(count);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<double>(logits[labels[index]]) / temperature;
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
    std::thread worker;
    std::string error;
};

namespace {

std::vector<double> evaluate_row(ibrh_job* job,
                                 const std::vector<llama_token>& tokens,
                                 std::size_t option_count,
                                 decider::native::AnswerType type) {
    ibrh_model* model = job->model;
    if (tokens.empty()) throw std::runtime_error("Decider prompt produced no tokens");
    if (tokens.size() > model->config.context_size)
        throw std::runtime_error("Decider prompt exceeds context_size");
    llama_memory_clear(llama_get_memory(model->context), true);
    std::size_t offset = 0;
    while (offset < tokens.size()) {
        if (job->cancel.load()) throw std::runtime_error("cancelled");
        const std::size_t count = std::min<std::size_t>(
            model->config.batch_size, tokens.size() - offset);
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token*>(tokens.data() + offset), static_cast<int>(count));
        const int result = llama_decode(model->context, batch);
        if (result != 0)
            throw std::runtime_error("llama.cpp prompt decode failed with code " +
                                     std::to_string(result));
        offset += count;
    }
    const float* logits = llama_get_logits_ith(model->context, -1);
    if (logits == nullptr) throw std::runtime_error("llama.cpp returned no answer-slot logits");
    return selected_softmax(logits, model->labels, option_count,
                            temperature_for(model->config, type));
}

void run_job(ibrh_job* job, std::string request_text,
             char* output, std::size_t output_capacity) noexcept {
    try {
        job->state = IBRH_JOB_RUNNING;
        auto plan = decider::native::parse_system_one(
            request_text, job->model->config.isolated_levels);
        if (!plan.independent)
            throw std::invalid_argument(
                "the initial native harness supports independent=true requests only");
        if (plan.rows.size() > job->model->config.max_rows)
            throw std::invalid_argument("request expands beyond max_rows");

        std::vector<std::vector<llama_token>> token_rows;
        token_rows.reserve(plan.rows.size());
        for (const auto& row : plan.rows) {
            if (row.options.size() > 10)
                throw std::invalid_argument(
                    "the initial native harness does not yet support wide (>10 option) labels");
            auto tokens = tokenize(job->model->vocabulary,
                "Context:\n" + plan.rendered_state);
            if (tokens.size() > job->model->config.max_state_tokens)
                tokens.resize(job->model->config.max_state_tokens);
            auto suffix = tokenize(job->model->vocabulary,
                decider::native::plain_narrow_block(row, 0, false));
            tokens.insert(tokens.end(), suffix.begin(), suffix.end());
            token_rows.push_back(std::move(tokens));
        }

        std::vector<std::vector<double>> probabilities;
        probabilities.reserve(plan.rows.size());
        for (std::size_t index = 0; index < plan.rows.size(); ++index) {
            if (job->cancel.load()) {
                job->state = IBRH_JOB_CANCELLED;
                return;
            }
            probabilities.push_back(evaluate_row(
                job, token_rows[index], plan.rows[index].options.size(),
                plan.rows[index].temperature_type));
        }
        const auto response = decider::native::assemble_response(
            plan, probabilities, job->model->config.model_name,
            unique_tokens(token_rows));
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
        for (char label = 'A'; label <= 'J'; ++label) {
            const auto encoded = tokenize(model->vocabulary, std::string(1, label));
            if (encoded.size() != 1) {
                llama_model_free(model->model);
                model->model = nullptr;
                throw std::runtime_error("Decider label does not map to one GGUF token");
            }
            model->labels.push_back(encoded.front());
        }
        llama_context_params context = llama_context_default_params();
        context.n_ctx = model->config.context_size;
        context.n_batch = model->config.batch_size;
        context.n_ubatch = model->config.batch_size;
        context.n_seq_max = 1;
        context.n_outputs_max = 1;
        context.n_outputs_max_per_seq = 1;
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
