#include "opengl.h"

GLuint compile_shader(GLenum type, std::initializer_list<std::string_view> sources)
{
    constexpr size_t max_sources = 16;
    const GLchar * strings[max_sources] {};
    GLint lengths[max_sources] {};
    GLsizei count = 0;
    for(auto source : sources)
    {
        strings[count] = source.data();
        lengths[count] = source.size();
        if(++count == max_sources) break;
    }

    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, count, strings, lengths);
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


std::string_view preamble = "#version 450\n";

std::string_view pbr_lighting = R"(
const float PI = 3.14159265359;

// This structure contains the essential information about a fragment to compute lighting for it
struct pbr_surface
{
    vec3 normal_vec;        // unit vector perpendicular to the surface
    vec3 eye_vec;           // unit vector pointing from the surface to the viewer
    float n_dot_v;          // max(dot(normal_vec, eye_vec), 0)
    vec3 diffuse_albedo;    // (1-metalness) * albedo/PI
    vec3 base_reflectivity; // F0
    float alpha;            // roughness^2
    float k;                // computed different for direct and indirect lighting
};

// This function computes the contribution of a single light to a fragment
vec3 compute_contribution(pbr_surface surf, vec3 light_vec, vec3 radiance)
{
    // Compute half vector and precompute some dot products
    vec3 half_vec = normalize(surf.eye_vec + light_vec);
    float n_dot_l = max(dot(surf.normal_vec, light_vec), 0);
    float n_dot_h = max(dot(surf.normal_vec, half_vec), 0);    
    float v_dot_h = max(dot(surf.eye_vec, half_vec), 0);

    // Evaluate Trowbridge-Reitz GGX normal distribution function
    float denom = n_dot_h*n_dot_h*(surf.alpha*surf.alpha-1) + 1;
    denom = PI * denom * denom;
    float D = (surf.alpha*surf.alpha) / denom;

    // Evaluate Smith's Schlick-GGX geometry function
    float ggx1 = n_dot_l / (n_dot_l*(1-surf.k) + surf.k);
    float ggx2 = surf.n_dot_v / (surf.n_dot_v*(1-surf.k) + surf.k);
    float G = ggx1 * ggx2;

    // Evaluate Fresnel-Schlick approximation to Fresnel equation
    vec3 F = surf.base_reflectivity + (1-surf.base_reflectivity) * pow(1-v_dot_h, 5);

    // Evaluate Cook-Torrance specular BRDF
    vec3 specular = (D * G * F) / (4 * surf.n_dot_v * n_dot_l + 0.001);  

    // Compute diffuse contribution
    vec3 diffuse = (1-F) * surf.diffuse_albedo;

    // Return total contribution from this light
    return (diffuse + specular) * radiance * n_dot_l;
}

// This function computes the full lighting to apply to a single fragment
uniform vec3 u_eye_position;
vec3 compute_lighting(vec3 position, vec3 normal, vec3 albedo, float roughness, float metalness, float ambient_occlusion)
{
    pbr_surface surf;
    surf.normal_vec = normalize(normal);
    surf.eye_vec = normalize(u_eye_position - position);
    surf.n_dot_v = max(dot(surf.normal_vec, surf.eye_vec), 0);
    surf.base_reflectivity = mix(vec3(0.04), albedo, metalness);
    surf.diffuse_albedo = (1-metalness) * albedo/PI;
    surf.alpha = roughness*roughness;
    surf.k = (roughness+1)*(roughness+1)/8;

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
        light += compute_contribution(surf, L, radiance);
    }
    return light;
}
)";