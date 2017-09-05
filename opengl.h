#ifndef OPENGL_H
#define OPENGL_H

#include <vector>
#include <string_view>

#include "3rdparty/linalg.h"
using namespace linalg::aliases;

#define GLEW_STATIC
#include "3rdparty/glew/include/GL/glew.h"
#include "3rdparty/glfw/include/GLFW/glfw3.h"
#pragma comment(lib, "opengl32.lib")

extern std::string_view preamble;
extern std::string_view pbr_lighting;

GLuint compile_shader(GLenum type, std::initializer_list<std::string_view> sources);
GLuint link_program(std::initializer_list<GLuint> shader_stages);

class pbr_tools
{
    GLuint spheremap_skybox_prog;
    GLuint cubemap_skybox_prog;
    GLuint cubemap_convolution_prog;
    GLuint prefilter_prog;
    GLuint brdf_integration_prog;
public:
    pbr_tools();

    GLuint convert_spheremap_to_cubemap(GLenum internal_format, GLsizei width, GLuint spheremap) const;
    GLuint compute_irradiance_map(GLuint cubemap) const;
    GLuint compute_reflectance_map(GLuint cubemap) const;
    GLuint compute_brdf_integration_map() const;

    void draw_skybox(GLuint cubemap, const float4x4 & view_proj_matrix) const;
};

#endif