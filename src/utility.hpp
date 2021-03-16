#ifndef VULKAN_SANDBOX_UTILITY_HPP
#define VULKAN_SANDBOX_UTILITY_HPP

#include <cstddef>
#include <fstream>
#include <vector>

inline std::vector<std::byte> read_file(const char *filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);

    if (!file)
        throw std::runtime_error("Could not open file!");

    const auto end = file.tellg();
    file.seekg(0, std::ios::beg);
    const auto start = file.tellg();

    std::vector<std::byte> buffer;

    const auto size = static_cast<std::size_t>(end - start);

    if (size == 0)
        return buffer;

    buffer.resize(size);
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));

    return buffer;
}

#endif // VULKAN_SANDBOX_UTILITY_HPP
