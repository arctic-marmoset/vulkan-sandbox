#ifndef VULKAN_RENDERER_CAMERA_HPP
#define VULKAN_RENDERER_CAMERA_HPP

#include "Utility.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>

class Camera
{
public:
    explicit Camera() = default;

    explicit Camera(
        glm::vec3 position,
        glm::vec3 lookDirection,
        float vFov,
        float aspect,
        float zNear
    )
        : m_Position(position)
        , m_LookDirection(lookDirection)
        , m_Fov(vFov)
        , m_Aspect(aspect)
        , m_NearClip(zNear)
    {
    }

    void Update(float deltaTime)
    {
        if (m_IsUpdateRequired)
        {
            m_IsUpdateRequired = false;

            const float lookAmount = deltaTime * m_LookSensitivity;
            const float moveAmount = deltaTime * m_MoveSpeed;

            m_Yaw += lookAmount * m_DeltaYaw;
            m_Pitch = glm::mod(m_Pitch + (lookAmount * m_DeltaPitch), 89.0F);
            m_Position += moveAmount * m_DeltaPosition;
            UpdateVectors();
            m_View = CalculateView();

            m_DeltaPitch = 0.0F;
            m_DeltaYaw = 0.0F;
            m_DeltaPosition = { };
        }
    }

    [[nodiscard]]
    glm::mat4 GetView() const
    {
        return m_View;
    }

    void ModPitch(float delta)
    {
        m_DeltaPitch = delta;
        m_IsUpdateRequired = true;
    }

    void ModYaw(float delta)
    {
        m_DeltaYaw = delta;
        m_IsUpdateRequired = true;
    }

    void Translate(glm::vec3 delta)
    {
        m_DeltaPosition = delta;
    }

private:
    void UpdateVectors()
    {
        const glm::vec3 lookDirection = {
            glm::cos(m_Yaw) * glm::cos(m_Pitch),
            glm::sin(m_Pitch),
            glm::sin(m_Yaw) * glm::cos(m_Yaw),
        };

        m_LookDirection = glm::normalize(lookDirection);
        m_Right = glm::normalize(glm::cross(m_LookDirection, WorldUp));
        m_Up = glm::normalize(glm::cross(m_Right, m_LookDirection));
    }

    glm::mat4 CalculateView()
    {
        return glm::lookAt(m_Position, m_Position + m_LookDirection, m_Up);
    }

private:
    glm::vec3 m_Position      = { };
    glm::vec3 m_LookDirection = { 0.0F, 0.0F, 1.0F };
    glm::vec3 m_Right         = glm::cross(m_LookDirection, WorldUp);
    glm::vec3 m_Up            = glm::cross(m_Right, m_LookDirection);

    glm::vec3 m_DeltaPosition = { };

    float m_Pitch = glm::angle(glm::vec3(0.0F, 0.0F, 1.0F), m_LookDirection);
    float m_Yaw   = glm::angle(glm::vec3(1.0F, 0.0F, 0.0F), m_LookDirection);

    float m_DeltaPitch = 0.0F;
    float m_DeltaYaw = 0.0F;

    glm::mat4 m_View = CalculateView();

    float m_Fov = glm::radians(59.0F);
    float m_Aspect = 16.0F / 9.0F;
    float m_NearClip = 0.01F;

    float m_LookSensitivity = 1.0F;
    float m_MoveSpeed = 1.0F;

    bool m_IsUpdateRequired = false;
};

#endif // !VULKAN_RENDERER_CAMERA_HPP
