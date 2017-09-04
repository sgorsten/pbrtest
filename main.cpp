#define GLEW_STATIC
#include "3rdparty/glew/include/GL/glew.h"
#include "3rdparty/glfw/include/GLFW/glfw3.h"
#pragma comment(lib, "opengl32.lib")

#include <chrono>
using hr_clock = std::chrono::high_resolution_clock;

#include "3rdparty/linalg.h"
using namespace linalg::aliases;

struct camera
{
    float3 position;
    float pitch=0, yaw=0;

    float4 get_orientation() const { return qmul(rotation_quat(float3{0,1,0}, yaw), rotation_quat(float3{1,0,0}, pitch)); }
    float4x4 get_view_matrix() const { return inverse(pose_matrix(get_orientation(), position)); }

    void move_local(const float3 & displacement) { position += qrot(get_orientation(), displacement); }
};

int main()
{
    glfwInit();

    auto win = glfwCreateWindow(1280, 720, "PBR Test", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glewInit();

    // Set up a right-handed, x-right, y-down, z-forward coordinate system with a 0-to-1 depth buffer
    glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW); // Still actually counter-clockwise

    const float cam_speed = 4;
    camera cam {{0,0,-5}};

    double2 prev_cursor;
    glfwGetCursorPos(win, &prev_cursor.x, &prev_cursor.y);
    auto t0 = hr_clock::now();
    while(!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        double2 cursor;
        glfwGetCursorPos(win, &cursor.x, &cursor.y);
        if(glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT))
        {
            cam.yaw += static_cast<float>(cursor.x - prev_cursor.x) * 0.01f;
            cam.pitch += static_cast<float>(prev_cursor.y - cursor.y) * 0.01f;
        }
        prev_cursor = cursor;

        const auto t1 = hr_clock::now();
        const float timestep = std::chrono::duration<float>(t1-t0).count();
        t0 = t1;

        float3 move;
        if(glfwGetKey(win, GLFW_KEY_W)) move.z += 1;
        if(glfwGetKey(win, GLFW_KEY_A)) move.x -= 1;
        if(glfwGetKey(win, GLFW_KEY_S)) move.z -= 1;
        if(glfwGetKey(win, GLFW_KEY_D)) move.x += 1;
        if(length(move) > 0) cam.move_local(normalize(move) * (cam_speed * timestep));

        const float4x4 view_matrix = cam.get_view_matrix();
        const float4x4 proj_matrix = linalg::perspective_matrix(1.0f, (float)1280/720, 0.1f, 10.0f, linalg::pos_z, linalg::zero_to_one);
        const float4x4 view_proj_matrix = mul(proj_matrix, view_matrix);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glLoadIdentity();
        glLoadMatrixf(&view_proj_matrix[0][0]);

        glBegin(GL_TRIANGLES);
        glVertex3f(-0.5f,+0.5f,0);
        glVertex3f(+0.5f,+0.5f,0);        
        glVertex3f( 0.0f,-0.5f,0);
        glEnd();

        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
}