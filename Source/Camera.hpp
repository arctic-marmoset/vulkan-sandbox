#ifndef VULKAN_RENDERER_CAMERA_HPP
#define VULKAN_RENDERER_CAMERA_HPP

#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>

class Camera
{
public:
    glm::vec3 GetPosition() const
    {
        return m_Position;
    }

    glm::vec3 GetLookDirection() const
    {
        return m_Forward;
    }

    void ResetAsync()
    {
        m_ShouldReset = true;
    }

    void MoveForwardAsync()
    {
        m_Movement.Forward = true;
    }

    void MoveBackwardAsync()
    {
        m_Movement.Backward = true;
    }

    void StrafeLeftAsync()
    {
        m_Movement.Left = true;
    }

    void StrafeRightAsync()
    {
        m_Movement.Right = true;
    }

    void SetShiftSpeedAsync()
    {
        m_Movement.Shift = true;
    }

    void SetAltSpeedAsync()
    {
        m_Movement.Alt = true;
    }

    glm::mat4 GetViewMatrix() const
    {
        return glm::lookAt(m_Position, m_Position + m_Forward, m_Up);
    }

    void Update(float deltaTime, float deltaX, float deltaY);

private:
    glm::vec3 m_Forward = { 0.0F, 0.0F, 1.0F };
    glm::vec3 m_Right = { 1.0F, 0.0F, 0.0F };
    glm::vec3 m_Up = { 0.0F, -1.0F, 0.0F };

    glm::vec3 m_Position = { 0.0F, 0.0F, -1.0F };

    struct
    {
        bool Forward;
        bool Backward;
        bool Left;
        bool Right;

        bool Shift;
        bool Alt;

        void Normalize()
        {
            if (Forward && Backward)
            {
                Forward = false;
                Backward = false;
            }

            if (Left && Right)
            {
                Left = false;
                Right = false;
            }

            if (Shift & Alt)
            {
                Shift = false;
                Alt = false;
            }
        }

        bool Any() const
        {
            return Forward || Backward || Left || Right;
        }
    } m_Movement = { };

    float m_Yaw   = glm::half_pi<float>();
    float m_Pitch = 0.0F;

    bool m_ShouldReset = false;
};

#endif // !VULKAN_RENDERER_CAMERA_HPP
