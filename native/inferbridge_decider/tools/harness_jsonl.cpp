#include <inferbridge/inferbridge_harness.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

ibrh_string_view view(const std::string& value) {
    return {value.data(), value.size()};
}

std::string last_error(const ibrh_api& api, const void* object) {
    std::size_t required = 0;
    api.get_last_error(object, nullptr, 0, &required);
    std::string message(required == 0 ? 1 : required, '\0');
    api.get_last_error(object, message.data(), message.size(), &required);
    if (!message.empty() && message.back() == '\0') message.pop_back();
    return message;
}

void check(const ibrh_api& api, ibrh_result result, const void* object,
           const char* operation) {
    if (result == IBRH_OK) return;
    throw std::runtime_error(std::string(operation) + " failed (" +
                             std::to_string(result) + "): " + last_error(api, object));
}

ibrh_resource host_json(void* data, std::size_t size, std::uint32_t access) {
    ibrh_resource resource{};
    resource.struct_size = sizeof(resource);
    resource.api_version = IBRH_CURRENT_API_VERSION;
    resource.domain = IBRH_RESOURCE_DOMAIN_HOST;
    resource.kind = IBRH_RESOURCE_KIND_BUFFER;
    resource.access = access;
    resource.pixel_format = IBRH_PAYLOAD_UTF8_JSON;
    resource.width = static_cast<std::uint32_t>(size);
    resource.height = 1;
    resource.depth = 1;
    resource.row_stride_bytes = static_cast<std::uint32_t>(size);
    resource.native_handle_type = IBRH_NATIVE_HANDLE_HOST_POINTER;
    resource.byte_size = size;
    resource.native_handle = static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(data));
    return resource;
}

ibrh_transfer_binding binding(ibrh_resource resource) {
    ibrh_transfer_binding result{};
    result.struct_size = sizeof(result);
    result.api_version = IBRH_CURRENT_API_VERSION;
    result.resource = resource;
    result.synchronization.struct_size = sizeof(result.synchronization);
    result.synchronization.api_version = IBRH_CURRENT_API_VERSION;
    result.synchronization.kind = IBRH_SYNC_NONE;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: decider_native_jsonl MODEL.gguf [CPU|VULKAN] "
                     "[--model-parameters JSON] [--ready] [--timing]\n";
        return 2;
    }
    try {
        const std::string model_path = argv[1];
        std::string backend = "CPU";
        std::string model_parameters;
        bool timing = false;
        bool ready = false;
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--timing") {
                timing = true;
            } else if (argument == "--ready") {
                ready = true;
            } else if (argument == "--model-parameters" && index + 1 < argc) {
                model_parameters = argv[++index];
            } else if (argument == "CPU" || argument == "VULKAN") {
                backend = argument;
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }

        ibrh_api api{};
        if (ibrh_get_api(IBRH_CURRENT_API_VERSION, sizeof(api), &api) != IBRH_OK)
            throw std::runtime_error("could not obtain InferBridge harness API");

        ibrh_runtime_create_request runtime_request{};
        runtime_request.struct_size = sizeof(runtime_request);
        runtime_request.api_version = IBRH_CURRENT_API_VERSION;
        runtime_request.backend = view(backend);
        ibrh_runtime* runtime = nullptr;
        check(api, api.runtime_create(sizeof(runtime_request), &runtime_request, &runtime),
              nullptr, "runtime_create");

        ibrh_model* model = nullptr;
        ibrh_job* job = nullptr;
        try {
            ibrh_model_load_request model_request{};
            model_request.struct_size = sizeof(model_request);
            model_request.api_version = IBRH_CURRENT_API_VERSION;
            model_request.model_path = view(model_path);
            model_request.parameters_json = view(model_parameters);
            const auto load_started = std::chrono::steady_clock::now();
            check(api, api.model_load(runtime, sizeof(model_request), &model_request, &model),
                  runtime, "model_load");
            const auto load_finished = std::chrono::steady_clock::now();

            ibrh_model_io_descriptor io{};
            check(api, api.model_describe_io(model, sizeof(io), &io), model,
                  "model_describe_io");
            if (io.input_count != 1 || io.output_count != 1)
                throw std::runtime_error("JSONL mode requires a text-only model interface");
            if (timing) {
                const auto load_ms = std::chrono::duration<double, std::milli>(
                    load_finished - load_started).count();
                std::cerr << "TIMING model_load_ms=" << load_ms << '\n';
            }
            if (ready) std::cout << "{\"ready\":true}" << '\n' << std::flush;

            std::string request_json;
            std::uint64_t request_index = 0;
            while (std::getline(std::cin, request_json)) {
                if (request_json.empty()) continue;
                ++request_index;
                const auto input_resource = host_json(
                    request_json.data(), request_json.size(), IBRH_RESOURCE_ACCESS_READ);
                ibrh_output_plan_request plan{};
                plan.struct_size = sizeof(plan);
                plan.api_version = IBRH_CURRENT_API_VERSION;
                plan.inputs = &input_resource;
                plan.input_count = 1;
                ibrh_port_descriptor output_port{};
                check(api, api.model_plan_outputs(model, sizeof(plan), &plan, 1,
                                                  &output_port),
                      model, "model_plan_outputs");

                std::vector<char> output(output_port.width, '\0');
                auto input_binding = binding(input_resource);
                auto output_binding = binding(host_json(
                    output.data(), output.size(), IBRH_RESOURCE_ACCESS_WRITE));
                ibrh_submit_request submit{};
                submit.struct_size = sizeof(submit);
                submit.api_version = IBRH_CURRENT_API_VERSION;
                submit.inputs = &input_binding;
                submit.input_count = 1;
                submit.outputs = &output_binding;
                submit.output_count = 1;
                submit.source_frame_id = request_index;
                const auto inference_started = std::chrono::steady_clock::now();
                check(api, api.submit(model, sizeof(submit), &submit, &job), model,
                      "submit");

                for (;;) {
                    ibrh_job_status status{};
                    check(api, api.job_poll(job, sizeof(status), &status), job,
                          "job_poll");
                    if (status.state == IBRH_JOB_COMPLETE) break;
                    if (status.state == IBRH_JOB_FAILED ||
                        status.state == IBRH_JOB_CANCELLED)
                        throw std::runtime_error(
                            "native inference failed: " + last_error(api, job));
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                const auto inference_finished = std::chrono::steady_clock::now();
                std::cout << output.data() << '\n' << std::flush;
                if (timing) {
                    const auto inference_ms = std::chrono::duration<double, std::milli>(
                        inference_finished - inference_started).count();
                    std::cerr << "TIMING request=" << request_index
                              << " inference_ms=" << inference_ms << '\n';
                }
                api.job_release(job);
                job = nullptr;
            }

            api.model_unload(model);
            model = nullptr;
            api.runtime_destroy(runtime);
            return 0;
        } catch (...) {
            if (job != nullptr) api.job_release(job);
            if (model != nullptr) api.model_unload(model);
            api.runtime_destroy(runtime);
            throw;
        }
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
