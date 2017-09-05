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
uniform sampler2D u_texture;
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
uniform samplerCube u_texture;
layout(location=0) in vec3 direction;
layout(location=0) out vec4 f_color;
void main()
{
    f_color = textureLod(u_texture, direction, 1.2);
})";

constexpr char cubemap_convolution_frag_shader_source[] = R"(
uniform samplerCube u_texture;
layout(location=0) in vec3 direction;
layout(location=0) out vec4 f_color;
void main()
{
    vec3 normal = normalize(direction);
    vec3 up = vec3(0,1,0);
    vec3 right = cross(up, normal);
    up = cross(normal, right);

    float sampleDelta = 0.01, nrSamples = 0; 
    vec3 irradiance = vec3(0);
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            vec3 tangentSample = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal; 

            irradiance += texture(u_texture, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }

    f_color = vec4(PI * irradiance / nrSamples, 1);

})";

constexpr char prefilter_frag_shader_source[] = R"(
uniform samplerCube u_texture;
uniform float u_roughness;
layout(location=0) in vec3 direction;
layout(location=0) out vec4 f_color;

float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}  
  
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness*roughness;
	
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

void main()
{		
    vec3 N = normalize(direction);    
    vec3 R = N;
    vec3 V = R;

    const uint SAMPLE_COUNT = 1024u;
    float totalWeight = 0.0;   
    vec3 prefilteredColor = vec3(0.0);     
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H  = ImportanceSampleGGX(Xi, N, u_roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0)
        {
            prefilteredColor += texture(u_texture, L).rgb * NdotL;
            totalWeight      += NdotL;
        }
    }

    f_color = vec4(prefilteredColor / totalWeight, 1);
})";

constexpr char brdf_integration_vert_shader_source[] = R"(
layout(location=0) in vec2 v_position;
layout(location=1) in vec2 v_texcoords;
layout(location=0) out vec2 texcoords;
void main()
{
    texcoords = v_texcoords;
    gl_Position = vec4(v_position,0,1);
})";

constexpr char brdf_integration_frag_shader_source[] = R"(
layout(location=0) in vec2 texcoords;
layout(location=0) out vec4 f_color;

float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}  
  
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness*roughness;
	
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec2 IntegrateBRDF(float NdotV, float roughness)
{
    vec3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    vec3 N = vec3(0.0, 0.0, 1.0);

    const uint SAMPLE_COUNT = 1024u;
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H  = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
}
void main() 
{
    f_color = vec4(IntegrateBRDF(texcoords.x, texcoords.y), 0, 1);
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

template<class F> GLuint render_cubemap(GLsizei levels, GLenum internal_format, GLsizei width, F draw_face)
{
    GLuint cubemap;
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &cubemap);
    glTextureStorage2D(cubemap, levels, internal_format, width, width);
    glTextureParameteri(cubemap, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(cubemap, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(cubemap, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTextureParameteri(cubemap, GL_TEXTURE_MIN_FILTER, levels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTextureParameteri(cubemap, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint fbo;
    glCreateFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    for(GLint mip=0; mip<levels; ++mip)
    {
        glViewport(0, 0, width, width);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X, cubemap, mip); draw_face(float4x4{{0,0,+1,0},{0,+1,0,0},{-1,0,0,0},{0,0,0,1}}, mip);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, cubemap, mip); draw_face(float4x4{{0,0,-1,0},{0,+1,0,0},{+1,0,0,0},{0,0,0,1}}, mip);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, cubemap, mip); draw_face(float4x4{{+1,0,0,0},{0,0,+1,0},{0,-1,0,0},{0,0,0,1}}, mip);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, cubemap, mip); draw_face(float4x4{{+1,0,0,0},{0,0,-1,0},{0,+1,0,0},{0,0,0,1}}, mip);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, cubemap, mip); draw_face(float4x4{{+1,0,0,0},{0,+1,0,0},{0,0,+1,0},{0,0,0,1}}, mip);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, cubemap, mip); draw_face(float4x4{{-1,0,0,0},{0,+1,0,0},{0,0,-1,0},{0,0,0,1}}, mip);
        width = std::max(width/2, 1);
    }
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    return cubemap; 
}

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
    auto cubemap_convolution_frag_shader = compile_shader(GL_FRAGMENT_SHADER, {preamble, cubemap_convolution_frag_shader_source});
    auto cubemap_convolution_prog = link_program({skybox_vert_shader, cubemap_convolution_frag_shader});
    auto prefilter_frag_shader = compile_shader(GL_FRAGMENT_SHADER, {preamble, prefilter_frag_shader_source});
    auto prefilter_prog = link_program({skybox_vert_shader, prefilter_frag_shader});

    auto brdf_integration_vert_shader = compile_shader(GL_VERTEX_SHADER, {preamble, brdf_integration_vert_shader_source});
    auto brdf_integration_frag_shader = compile_shader(GL_FRAGMENT_SHADER, {preamble, brdf_integration_frag_shader_source});
    auto brdf_integration_prog = link_program({brdf_integration_vert_shader, brdf_integration_frag_shader});

    // Set up a right-handed, x-right, y-down, z-forward coordinate system with a 0-to-1 depth buffer
    glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW); // Still actually counter-clockwise
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

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
    const GLuint environment_cubemap = render_cubemap(1, GL_RGBA16F, 1024, [&](const float4x4 & view_proj_matrix, int mip)
    {
        glUseProgram(spheremap_skybox_prog);
        glUniformMatrix4fv(glGetUniformLocation(spheremap_skybox_prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();
    });

    const GLuint irradiance_cubemap = render_cubemap(1, GL_RGBA16F, 32, [&](const float4x4 & view_proj_matrix, int mip)
    {
        glUseProgram(cubemap_convolution_prog);
        glUniformMatrix4fv(glGetUniformLocation(cubemap_convolution_prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glBindTexture(GL_TEXTURE_CUBE_MAP, environment_cubemap);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();
    });

    const GLuint prefiltered_cubemap = render_cubemap(5, GL_RGBA16F, 128, [&](const float4x4 & view_proj_matrix, int mip)
    {
        glUseProgram(prefilter_prog);
        glUniformMatrix4fv(glGetUniformLocation(prefilter_prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glUniform1f(glGetUniformLocation(prefilter_prog, "u_roughness"), mip/4.0f);
        glBindTexture(GL_TEXTURE_CUBE_MAP, environment_cubemap);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();
    });
    
    GLuint brdf_integration_map;
    glCreateTextures(GL_TEXTURE_2D, 1, &brdf_integration_map);
    glTextureStorage2D(brdf_integration_map, 1, GL_RG16F, 512, 512);
    glTextureParameteri(brdf_integration_map, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(brdf_integration_map, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(brdf_integration_map, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(brdf_integration_map, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint fbo;
    glCreateFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdf_integration_map, 0);
    glViewport(0,0,512,512);

    glUseProgram(brdf_integration_prog);
    glBegin(GL_QUADS);
    glVertexAttrib2f(1, 0, 0); glVertex2f(-1, -1);
    glVertexAttrib2f(1, 0, 1); glVertex2f(-1, +1);
    glVertexAttrib2f(1, 1, 1); glVertex2f(+1, +1);
    glVertexAttrib2f(1, 1, 0); glVertex2f(+1, -1);
    glEnd();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_DEPTH_TEST);

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
        glBindTexture(GL_TEXTURE_CUBE_MAP, glfwGetKey(win, GLFW_KEY_C) ? prefiltered_cubemap : environment_cubemap);
        glDepthMask(GL_FALSE);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();

        // Render spheres
        glDepthMask(GL_TRUE);

        for(int i : {0,1}) glEnableVertexAttribArray(i);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), &sphere_verts.front().position[0]);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), &sphere_verts.front().normal[0]);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, irradiance_cubemap);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_CUBE_MAP, prefiltered_cubemap);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, brdf_integration_map);
        glActiveTexture(GL_TEXTURE0);

        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "u_irradiance_map"), 0);
        glUniform1i(glGetUniformLocation(prog, "u_prefiltered_map"), 1);
        glUniform1i(glGetUniformLocation(prog, "u_brdf_integration_map"), 2);
        glUniformMatrix4fv(glGetUniformLocation(prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glUniform3fv(glGetUniformLocation(prog, "u_eye_position"), 1, &cam.position[0]);

        const float3 albedo {1,1,1}; //0,0};
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