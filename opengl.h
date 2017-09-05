#ifndef OPENGL_H
#define OPENGL_H

#define GLEW_STATIC
#include "3rdparty/glew/include/GL/glew.h"
#include "3rdparty/glfw/include/GLFW/glfw3.h"
#pragma comment(lib, "opengl32.lib")

#include <vector>
#include <string_view>

extern std::string_view preamble;
extern std::string_view pbr_lighting;

GLuint compile_shader(GLenum type, std::initializer_list<std::string_view> sources);
GLuint link_program(std::initializer_list<GLuint> shader_stages);

#endif