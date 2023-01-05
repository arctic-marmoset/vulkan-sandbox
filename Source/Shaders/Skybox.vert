#version 450

layout(set = 0, binding = 0) uniform Ubo
{
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outCubeCoord;

void main()
{
    vec4 position = ubo.proj * mat4(mat3(ubo.view)) * vec4(inPosition, 1.0);
    position.z = 0.0;
    gl_Position = position;
    outCubeCoord = inPosition;
}
