#include <inferbridge/inferbridge_harness.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ibrh_string_view literal(const char* value) {
    return {value, std::strlen(value)};
}

}  // namespace

int main() {
    try {
        ibrh_api api{};
        require(ibrh_get_api(IBRH_CURRENT_API_VERSION, sizeof(api), &api) == IBRH_OK,
                "ibrh_get_api failed");
        require(api.struct_size == sizeof(api) && api.submit != nullptr,
                "native harness API table is incomplete");

        ibrh_capabilities capabilities{};
        require(api.query_capabilities(sizeof(capabilities), &capabilities) == IBRH_OK,
                "query_capabilities failed");
        require(capabilities.maximum_inputs == 1 && capabilities.maximum_outputs == 1,
                "Decider must expose one JSON input and output");
        require((capabilities.flags & IBRH_CAP_HOST_MEMORY) != 0,
                "Decider must support host-memory JSON");

        ibrh_runtime_create_request request{};
        request.struct_size = sizeof(request);
        request.api_version = IBRH_CURRENT_API_VERSION;
        request.backend = literal("CPU");
        ibrh_runtime* runtime = nullptr;
        require(api.runtime_create(sizeof(request), &request, &runtime) == IBRH_OK &&
                    runtime != nullptr,
                "CPU runtime creation failed");
        api.runtime_destroy(runtime);
        std::cout << "DECIDER_NATIVE_ABI_OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
