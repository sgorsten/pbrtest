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

class gl_program
{
    GLuint program = 0;
public:
    gl_program() = default;
    gl_program(std::initializer_list<GLuint> shader_stages);
    gl_program(gl_program && r) : gl_program() { *this = std::move(r); }
    ~gl_program();

    void use() const { glUseProgram(program); }

    void bind_texture(GLint location, GLuint texture) const { GLint binding; glGetUniformiv(program, location, &binding); glBindTextureUnit(binding, texture); }
    void bind_texture(const char * name, GLuint texture) const { const GLint location = glGetUniformLocation(program, name); if(location >= 0) bind_texture(location, texture); }

    void uniform(GLint location, float scalar) { glProgramUniform1f(program, location, scalar); }
    void uniform(GLint location, const float2 & vec) { glProgramUniform2fv(program, location, 1, &vec[0]); }
    void uniform(GLint location, const float3 & vec) { glProgramUniform3fv(program, location, 1, &vec[0]); }
    void uniform(GLint location, const float4 & vec) { glProgramUniform4fv(program, location, 1, &vec[0]); }
    void uniform(GLint location, const float4x4 & mat) { glProgramUniformMatrix4fv(program, location, 1, GL_FALSE, &mat[0][0]); }
    template<class T> void uniform(const char * name, const T & value) { const GLint location = glGetUniformLocation(program, name); if(location < 0) return; uniform(location, value); }

    gl_program & operator = (gl_program && r) { std::swap(program, r.program); return *this; }
};

GLuint compile_shader(GLenum type, std::initializer_list<std::string_view> sources);

class pbr_tools
{
    mutable gl_program spheremap_skybox_prog;
    mutable gl_program cubemap_skybox_prog;
    mutable gl_program irradiance_prog;
    mutable gl_program reflectance_prog;
    mutable gl_program brdf_integration_prog;
public:
    pbr_tools();

    GLuint convert_spheremap_to_cubemap(GLenum internal_format, GLsizei width, GLuint spheremap) const;
    GLuint compute_irradiance_map(GLuint cubemap) const;
    GLuint compute_reflectance_map(GLuint cubemap) const;
    GLuint compute_brdf_integration_map() const;

    void draw_skybox(GLuint cubemap, const float4x4 & skybox_view_proj_matrix) const;
};

#endif