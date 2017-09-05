#include <vector>
#include <iostream>
#include <string_view>
#include <chrono>
using hr_clock = std::chrono::high_resolution_clock;

#include "3rdparty/linalg.h"
using namespace linalg::aliases;

#include "opengl.h"

constexpr char skybox_vert_shader_source[] = R"(
uniform mat4 u_view_proj_matrix;
layout(location=0) in vec3 v_direction;
layout(location=0) out vec3 direction;
void main()
{
    direction   = v_direction;
    gl_Position = u_view_proj_matrix * vec4(direction,1);
})";

constexpr char spheremap_skybox_frag_shader_source[] = R"(
layout(location=0) uniform sampler2D u_texture;
layout(location=0) in vec3 direction;
layout(location=0) out vec4 f_color;
vec2 compute_spherical_texcoords(vec3 direction)
{
    return vec2(atan(direction.x, direction.z)*0.1591549, asin(direction.y)*0.3183099 + 0.5);
}
void main()
{
    f_color = texture(u_texture, compute_spherical_texcoords(normalize(direction)));
})";

constexpr char cubemap_skybox_frag_shader_source[] = R"(
layout(location=0) uniform samplerCube u_texture;
layout(location=0) in vec3 direction;
layout(location=0) out vec4 f_color;
void main()
{
    f_color = texture(u_texture, direction);
})";

constexpr char vert_shader_source[] = R"(
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

constexpr char frag_shader_source[] = R"(
// Fragment shader for untextured PBR surface
uniform vec3 u_albedo;
uniform float u_roughness;
uniform float u_metalness;
uniform float u_ambient_occlusion;
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=0) out vec4 f_color;
void main() 
{ 
    // Compute our PBR lighting
    vec3 light = compute_lighting(position, normal, u_albedo, u_roughness, u_metalness, u_ambient_occlusion);

    // Apply simple tone mapping and write to fragment
    f_color = vec4(light / (light + 1), 1);
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

#define STB_IMAGE_IMPLEMENTATION
#include "3rdparty/stb_image.h"

int main() try
{
    auto sphere_verts = make_sphere(32,16,0.4f);
    constexpr float3 skybox_verts[]
    {
        {-1,-1,-1}, {-1,+1,-1}, {-1,+1,+1}, {-1,-1,+1},
        {+1,-1,-1}, {+1,-1,+1}, {+1,+1,+1}, {+1,+1,-1},
        {-1,-1,-1}, {-1,-1,+1}, {+1,-1,+1}, {+1,-1,-1},
        {-1,+1,-1}, {+1,+1,-1}, {+1,+1,+1}, {-1,+1,+1},
        {-1,-1,-1}, {+1,-1,-1}, {+1,+1,-1}, {-1,+1,-1},
        {-1,-1,+1}, {-1,+1,+1}, {+1,+1,+1}, {+1,-1,+1}
    };

    glfwInit();

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
    auto win = glfwCreateWindow(1280, 720, "PBR Test", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewInit();

    auto vert_shader = compile_shader(GL_VERTEX_SHADER, {preamble, vert_shader_source});
    auto frag_shader = compile_shader(GL_FRAGMENT_SHADER, {preamble, pbr_lighting, frag_shader_source});
    auto prog = link_program({vert_shader, frag_shader});

    auto skybox_vert_shader = compile_shader(GL_VERTEX_SHADER, {preamble, skybox_vert_shader_source});
    auto spheremap_skybox_frag_shader = compile_shader(GL_FRAGMENT_SHADER, {preamble, spheremap_skybox_frag_shader_source});
    auto spheremap_skybox_prog = link_program({skybox_vert_shader, spheremap_skybox_frag_shader});
    auto cubemap_skybox_frag_shader = compile_shader(GL_FRAGMENT_SHADER, {preamble, cubemap_skybox_frag_shader_source});
    auto cubemap_skybox_prog = link_program({skybox_vert_shader, cubemap_skybox_frag_shader});

    // Set up a right-handed, x-right, y-down, z-forward coordinate system with a 0-to-1 depth buffer
    glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW); // Still actually counter-clockwise
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_MULTISAMPLE);

    int width, height;
    float * pixels = stbi_loadf("monument-valley.hdr", &width, &height, nullptr, 3);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGB, GL_FLOAT, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    stbi_image_free(pixels);

    // Convert spheremap to cubemap
    GLuint fbo, cubemap;

    glGenTextures(1, &cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
    for(int i=0; i<6; ++i) glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, GL_RGB16F, 1024, 1024, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glUseProgram(spheremap_skybox_prog);

    constexpr std::pair<GLenum, float4x4> cubemap_faces[6]
    {
        {GL_TEXTURE_CUBE_MAP_POSITIVE_X, {{0,0,+1,0},{0,+1,0,0},{-1,0,0,0},{0,0,0,1}}},
        {GL_TEXTURE_CUBE_MAP_NEGATIVE_X, {{0,0,-1,0},{0,+1,0,0},{+1,0,0,0},{0,0,0,1}}},
        {GL_TEXTURE_CUBE_MAP_POSITIVE_Y, {{+1,0,0,0},{0,0,+1,0},{0,-1,0,0},{0,0,0,1}}},
        {GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, {{+1,0,0,0},{0,0,-1,0},{0,+1,0,0},{0,0,0,1}}},
        {GL_TEXTURE_CUBE_MAP_POSITIVE_Z, {{+1,0,0,0},{0,+1,0,0},{0,0,+1,0},{0,0,0,1}}},
        {GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, {{-1,0,0,0},{0,+1,0,0},{0,0,-1,0},{0,0,0,1}}},
    };
    for(auto p : cubemap_faces)
    {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, p.first, cubemap, 0);
        glViewport(0, 0, 1024, 1024);
        glUniformMatrix4fv(glGetUniformLocation(spheremap_skybox_prog, "u_view_proj_matrix"), 1, GL_FALSE, &p.second[0][0]);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Initialize camera
    const float cam_speed = 8;
    camera cam {{0,0,-8}};
    double2 prev_cursor;
    glfwGetCursorPos(win, &prev_cursor.x, &prev_cursor.y);
    auto t0 = hr_clock::now();
    while(!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        // Handle user input
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

        // Set up scene
        const float4x4 view_matrix = cam.get_view_matrix();
        const float4x4 proj_matrix = linalg::perspective_matrix(1.0f, (float)1280/720, 0.1f, 32.0f, linalg::pos_z, linalg::zero_to_one);
        const float4x4 view_proj_matrix = mul(proj_matrix, view_matrix);
        auto skybox_cam = cam;
        skybox_cam.position = {0,0,0};
        const float4x4 skybox_view_proj_matrix = mul(proj_matrix, skybox_cam.get_view_matrix());

        // Render skybox
        glfwGetFramebufferSize(win, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(cubemap_skybox_prog);
        glUniformMatrix4fv(glGetUniformLocation(cubemap_skybox_prog, "u_view_proj_matrix"), 1, GL_FALSE, &skybox_view_proj_matrix[0][0]);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
        glDepthMask(GL_FALSE);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();

        // Render spheres
        glDepthMask(GL_TRUE);

        for(int i : {0,1}) glEnableVertexAttribArray(i);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), &sphere_verts.front().position[0]);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), &sphere_verts.front().normal[0]);

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
                glDrawArrays(GL_QUADS, 0, sphere_verts.size());
            }
        }
        for(int i : {0,1}) glDisableVertexAttribArray(i);

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