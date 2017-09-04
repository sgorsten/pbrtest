#define GLEW_STATIC
#include "3rdparty/glew/include/GL/glew.h"
#include "3rdparty/glfw/include/GLFW/glfw3.h"
#pragma comment(lib, "opengl32.lib")

int main()
{
    glfwInit();

    auto win = glfwCreateWindow(1280, 720, "PBR Test", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glewInit();

    while(!glfwWindowShouldClose(win))
    {
        glfwPollEvents();


        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
}