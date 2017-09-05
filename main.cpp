#include <vector>
#include <iostream>
#include <string_view>
#include <chrono>
using hr_clock = std::chrono::high_resolution_clock;

#include "3rdparty/linalg.h"
using namespace linalg::aliases;

#define GLEW_STATIC
#include "3rdparty/glew/include/GL/glew.h"
#include "3rdparty/glfw/include/GLFW/glfw3.h"
#pragma comment(lib, "opengl32.lib")

constexpr char vert_shader_source[] = R"(#version 420
uniform mat4 u_view_proj_matrix;
uniform mat4 u_model_matrix;
layout(location=0) in vec3 v_position;
layout(location=1) in vec3 v_normal;
layout(location=0) out vec3 position;
layout(location=1) out vec3 normal;
void main()
{
    position    = (u_model_matrix * vec4(v_position,1)).xyz;
    normal      = (u_model_matrix * vec4(v_normal,0)).xyz;
    gl_Position = u_view_proj_matrix * vec4(position,1);
})";

constexpr char frag_shader_source[] = R"(#version 420
const float PI = 3.14159265359;

vec3 compute_contribution(vec3 albedo, float roughness, float metalness, vec3 F0, vec3 N, vec3 V, float NdotV, vec3 L, vec3 radiance)
{
    // Compute half vector and precompute some dot products
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0);
    float NdotH = max(dot(N, H), 0);    

    // Evaluate Trowbridge-Reitz GGX normal distribution function
    float alpha = roughness*roughness;
    float denom = NdotH*NdotH*(alpha*alpha-1) + 1;
    denom = PI * denom * denom;
    float D = (alpha*alpha) / denom;

    // Evaluate Smith's Schlick-GGX geometry function
    float k_direct = (roughness+1)*(roughness+1)/8;
    float ggx1 = NdotL / (NdotL*(1-k_direct) + k_direct);
    float ggx2 = NdotV / (NdotV*(1-k_direct) + k_direct);
    float G = ggx1 * ggx2;

    // Evaluate Fresnel-Schlick approximation to Fresnel equation
    vec3 F = F0 + (1-F0) * pow(1-max(dot(H, V), 0), 5);

    // Evaluate Cook-Torrance specular BRDF
    vec3 specular = (D * G * F) / (4 * NdotV * NdotL + 0.001);  

    // Compute diffuse contribution
    vec3 diffuse = (1-F) * (1-metalness) * albedo/PI;

    // Return total contribution from this light
    return (diffuse + specular) * radiance * NdotL;
}

uniform vec3 u_eye_position;
vec3 compute_lighting(vec3 position, vec3 normal, vec3 albedo, float roughness, float metalness, float ambient_occlusion)
{
    // Determine common lighting equation terms
    vec3 N = normalize(normal); 
    vec3 V = normalize(u_eye_position - position);
    float NdotV = max(dot(N, V), 0);
    vec3 F0 = mix(vec3(0.04), albedo, metalness);

    // Initialize ambient light amount
    vec3 light = vec3(0.03) * albedo * ambient_occlusion;

    // Add contributions from point lights
    vec3 light_positions[4] = {vec3(-2, -2, -8), vec3(2, -2, -8), vec3(2, 2, -8), vec3(-2, 2, -8)};
    vec3 light_colors[4] = {vec3(23.47, 21.31, 20.79), vec3(23.47, 21.31, 20.79), vec3(23.47, 21.31, 20.79), vec3(23.47, 21.31, 20.79)};
    for(int i=0; i<4; ++i)
    {
        vec3 L = normalize(light_positions[i] - position);
        float distance = length(light_positions[i] - position);
        vec3 radiance  = light_colors[i] / (distance * distance); 
        light += compute_contribution(albedo, roughness, metalness, F0, N, V, NdotV, L, radiance);
    }
    return light;
}

uniform vec3  u_albedo;
uniform float u_roughness;
uniform float u_metalness;
uniform float u_ambient_occlusion;
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;

void main() 
{ 
    // Compute our PBR lighting
    vec3 light = compute_lighting(position, normal, u_albedo, u_roughness, u_metalness, u_ambient_occlusion);

    // Apply simple tone mapping and write to fragment
    gl_FragColor = vec4(light / (light + 1), 1);
})";

struct camera
{
    float3 position;
    float pitch=0, yaw=0;

    float4 get_orientation() const { return qmul(rotation_quat(float3{0,1,0}, yaw), rotation_quat(float3{1,0,0}, pitch)); }
    float4x4 get_view_matrix() const { return inverse(pose_matrix(get_orientation(), position)); }

    void move_local(const float3 & displacement) { position += qrot(get_orientation(), displacement); }
};

struct vertex { float3 position, normal; };
std::vector<vertex> make_sphere(int slices, int stacks, float radius)
{
    const auto make_vertex = [slices, stacks, radius](int i, int j)
    {
        const float tau = 6.2831853f, longitude = i*tau/slices, latitude = (j-(stacks*0.5f))*tau/2/stacks;
        const float3 normal {cos(longitude)*cos(latitude), sin(latitude), sin(longitude)*cos(latitude)}; // Poles at +/-y
        return vertex{normal*radius, normal};
    };

    std::vector<vertex> vertices;
    for(int i=0; i<slices; ++i)
    {
        for(int j=0; j<stacks; ++j)
        {
            vertices.push_back(make_vertex(i,j));
            vertices.push_back(make_vertex(i,j+1));
            vertices.push_back(make_vertex(i+1,j+1));
            vertices.push_back(make_vertex(i+1,j));
        }
    }
    return vertices;
}

GLuint compile_shader(GLenum type, std::string_view source)
{
    const GLuint shader = glCreateShader(type);
    const GLchar * string = source.data();
    const GLint length = source.size();
    glShaderSource(shader, 1, &string, &length);
    glCompileShader(shader);

    GLint compile_status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
    if(compile_status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);

        std::vector<GLchar> info_log(info_log_length);
        glGetShaderInfoLog(shader, info_log.size(), nullptr, info_log.data());
        glDeleteShader(shader);
        throw std::runtime_error(info_log.data());
    }

    return shader;
}

GLuint link_program(std::initializer_list<GLuint> shader_stages)
{
    const GLuint program = glCreateProgram();
    for(auto shader : shader_stages) glAttachShader(program, shader);
    glLinkProgram(program);

    GLint link_status;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if(link_status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);

        std::vector<GLchar> info_log(info_log_length);
        glGetProgramInfoLog(program, info_log.size(), nullptr, info_log.data());
        glDeleteProgram(program);
        throw std::runtime_error(info_log.data());
    }

    return program;
}

int main() try
{
    auto sphere_verts = make_sphere(32,16,0.4f);

    glfwInit();

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
    auto win = glfwCreateWindow(1280, 720, "PBR Test", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewInit();

    auto vert_shader = compile_shader(GL_VERTEX_SHADER, vert_shader_source);
    auto frag_shader = compile_shader(GL_FRAGMENT_SHADER, frag_shader_source);
    auto prog = link_program({vert_shader, frag_shader});

    // Set up a right-handed, x-right, y-down, z-forward coordinate system with a 0-to-1 depth buffer
    glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW); // Still actually counter-clockwise
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_MULTISAMPLE);

    const float cam_speed = 8;
    camera cam {{0,0,-8}};

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
        const float4x4 proj_matrix = linalg::perspective_matrix(1.0f, (float)1280/720, 0.1f, 32.0f, linalg::pos_z, linalg::zero_to_one);
        const float4x4 view_proj_matrix = mul(proj_matrix, view_matrix);

        glClearColor(0.5f,0.5f,0.5f,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(glGetUniformLocation(prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glUniform3fv(glGetUniformLocation(prog, "u_eye_position"), 1, &cam.position[0]);

        const float3 albedo {1,0,0};
        glUniform3fv(glGetUniformLocation(prog, "u_albedo"), 1, &albedo[0]);
        glUniform1f(glGetUniformLocation(prog, "u_ambient_occlusion"), 1.0f);
        for(int i=0; i<7; ++i)
        {
            for(int j=0; j<7; ++j)
            {
                const float3 position {j-3.0f, i-3.0f, 0.0f};
                const float4x4 model_matrix = translation_matrix(position);
                glUniformMatrix4fv(glGetUniformLocation(prog, "u_model_matrix"), 1, GL_FALSE, &model_matrix[0][0]);

                glUniform1f(glGetUniformLocation(prog, "u_metalness"), 1-(i+0.5f)/7);
                glUniform1f(glGetUniformLocation(prog, "u_roughness"), (j+0.5f)/7);

                glEnableVertexAttribArray(0);
                glEnableVertexAttribArray(1);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), &sphere_verts.front().position[0]);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), &sphere_verts.front().normal[0]);
                glDrawArrays(GL_QUADS, 0, sphere_verts.size());
            }
        }       

        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return EXIT_SUCCESS;
}
catch(const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}