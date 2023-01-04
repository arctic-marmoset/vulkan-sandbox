#include "Camera.hpp"

void Camera::Update(float deltaTime, float deltaX, float deltaY)
{
    if (m_ShouldReset)
    {
        m_ShouldReset = false;
        *this = Camera();
        return;
    }

    constexpr float lookSensitivity = 0.005F;

    constexpr float pitchLimit = glm::half_pi<float>() - 0.01F;
    const float deltaYaw = lookSensitivity * -deltaX;
    const float deltaPitch = lookSensitivity * deltaY;

    m_Yaw = glm::mod(m_Yaw + deltaYaw, glm::two_pi<float>());
    m_Pitch = glm::clamp(m_Pitch + deltaPitch, -pitchLimit, pitchLimit);

    const glm::vec3 lookDirection = {
        glm::cos(m_Yaw) * glm::cos(m_Pitch),
        glm::sin(m_Pitch),
        glm::sin(m_Yaw) * glm::cos(m_Pitch),
    };

    m_Forward = glm::normalize(lookDirection);
    m_Right = glm::normalize(glm::cross(m_Forward, glm::vec3(0.0F, -1.0F, 0.0F)));
    m_Up = glm::normalize(glm::cross(m_Right, m_Forward));

    glm::vec3 moveDirection = { };

    m_Movement.Normalize();
    if (m_Movement.Forward)
    {
        moveDirection += m_Forward;
    }
    if (m_Movement.Backward)
    {
        moveDirection -= m_Forward;
    }
    if (m_Movement.Left)
    {
        moveDirection -= m_Right;
    }
    if (m_Movement.Right)
    {
        moveDirection += m_Right;
    }

    if (m_Movement.Any())
    {
        constexpr float moveSpeed = 5.0F;
        constexpr float shiftMoveSpeed = 10.0F;
        constexpr float altMoveSpeed = 2.0F;

        const float moveAmount = m_Movement.Shift
            ? deltaTime * shiftMoveSpeed
            : m_Movement.Alt
                ? deltaTime * altMoveSpeed
                : deltaTime * moveSpeed;

        m_Position += moveAmount * glm::normalize(moveDirection);
    }

    m_Movement = { };
}
