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


std::string_view preamble = "#version 450\nconst float PI = 3.14159265359;\n";

std::string_view pbr_lighting = R"(

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
uniform samplerCube u_irradiance_map;
uniform samplerCube u_prefiltered_map;
uniform sampler2D u_brdf_integration_map;

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
} 

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

    vec3 R = reflect(-surf.eye_vec, surf.normal_vec);   
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(u_prefiltered_map, R, roughness * MAX_REFLECTION_LOD).rgb;    

    vec3 F          = fresnelSchlickRoughness(surf.n_dot_v, surf.base_reflectivity, roughness); 
    vec2 envBRDF    = texture(u_brdf_integration_map, vec2(surf.n_dot_v, roughness)).rg;
    vec3 specular   = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    vec3 kS         = F;
    vec3 kD         = (1 - kS) * (1 - metalness);
    vec3 irradiance = texture(u_irradiance_map, surf.normal_vec).rgb;
    vec3 diffuse    = irradiance * albedo;
    vec3 ambient    = (kD * diffuse + specular) * ambient_occlusion; 
    vec3 light      = ambient;

    // Add contributions from point lights
    vec3 light_positions[4] = {vec3(-3, -3, 8), vec3(3, -3, 8), vec3(3, 3, 8), vec3(-3, 3, 8)};
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

constexpr char importance_sample_ggx[] = R"(
vec2 hammersley_sequence(uint i, uint N)
{
    // Evaluate Van Der Corpus sequence
    uint bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float radical_inverse = float(bits) * 2.3283064365386963e-10; // / 0x100000000

    return vec2(float(i)/float(N), radical_inverse);
}  
  
vec3 importance_sample_ggx(vec2 Xi, vec3 N, float roughness)
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
)";

constexpr char prefilter_frag_shader_source[] = R"(
uniform samplerCube u_texture;
uniform float u_roughness;
layout(location=0) in vec3 direction;
layout(location=0) out vec4 f_color;

void main()
{		
    vec3 N = normalize(direction);    
    vec3 R = N;
    vec3 V = R;

    const uint SAMPLE_COUNT = 1024u;   
    vec3 sum_color = vec3(0,0,0);
    float sum_weight = 0;     
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 Xi = hammersley_sequence(i, SAMPLE_COUNT);
        vec3 H  = importance_sample_ggx(Xi, N, u_roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float n_dot_l = dot(N, L);
        if(n_dot_l > 0)
        {
            sum_color += texture(u_texture, L).rgb * n_dot_l;
            sum_weight += n_dot_l;
        }
    }

    f_color = vec4(sum_color/sum_weight, 1);
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
        vec2 Xi = hammersley_sequence(i, SAMPLE_COUNT);
        vec3 H  = importance_sample_ggx(Xi, N, roughness);
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

pbr_tools::pbr_tools()
{
    GLuint skybox_vert_shader  = compile_shader(GL_VERTEX_SHADER, {preamble, skybox_vert_shader_source});
    spheremap_skybox_prog      = link_program({skybox_vert_shader, compile_shader(GL_FRAGMENT_SHADER, {preamble, spheremap_skybox_frag_shader_source})});
    cubemap_skybox_prog        = link_program({skybox_vert_shader, compile_shader(GL_FRAGMENT_SHADER, {preamble, cubemap_skybox_frag_shader_source})});
    cubemap_convolution_prog   = link_program({skybox_vert_shader, compile_shader(GL_FRAGMENT_SHADER, {preamble, cubemap_convolution_frag_shader_source})});
    prefilter_prog             = link_program({skybox_vert_shader, compile_shader(GL_FRAGMENT_SHADER, {preamble, importance_sample_ggx, prefilter_frag_shader_source})});
    brdf_integration_prog      = link_program({compile_shader(GL_VERTEX_SHADER, {preamble, brdf_integration_vert_shader_source}), 
                                               compile_shader(GL_FRAGMENT_SHADER, {preamble, importance_sample_ggx, brdf_integration_frag_shader_source})});
}

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

constexpr float3 skybox_verts[]
{
    {-1,-1,-1}, {-1,+1,-1}, {-1,+1,+1}, {-1,-1,+1},
    {+1,-1,-1}, {+1,-1,+1}, {+1,+1,+1}, {+1,+1,-1},
    {-1,-1,-1}, {-1,-1,+1}, {+1,-1,+1}, {+1,-1,-1},
    {-1,+1,-1}, {+1,+1,-1}, {+1,+1,+1}, {-1,+1,+1},
    {-1,-1,-1}, {+1,-1,-1}, {+1,+1,-1}, {-1,+1,-1},
    {-1,-1,+1}, {-1,+1,+1}, {+1,+1,+1}, {+1,-1,+1}
};

GLuint pbr_tools::convert_spheremap_to_cubemap(GLenum internal_format, GLsizei width, GLuint spheremap) const
{
    glUseProgram(spheremap_skybox_prog);
    glBindTexture(GL_TEXTURE_2D, spheremap);
    return render_cubemap(1, internal_format, width, [&](const float4x4 & view_proj_matrix, int mip)
    {        
        glUniformMatrix4fv(glGetUniformLocation(spheremap_skybox_prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();
    });
}

GLuint pbr_tools::compute_irradiance_map(GLuint cubemap) const
{
    return render_cubemap(1, GL_RGB16F, 32, [&](const float4x4 & view_proj_matrix, int mip)
    {
        glUseProgram(cubemap_convolution_prog);
        glUniformMatrix4fv(glGetUniformLocation(cubemap_convolution_prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();
    });
}

GLuint pbr_tools::compute_reflectance_map(GLuint cubemap) const
{
    return render_cubemap(5, GL_RGB16F, 128, [&](const float4x4 & view_proj_matrix, int mip)
    {
        glUseProgram(prefilter_prog);
        glUniformMatrix4fv(glGetUniformLocation(prefilter_prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
        glUniform1f(glGetUniformLocation(prefilter_prog, "u_roughness"), mip/4.0f);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
        glBegin(GL_QUADS);
        for(auto & v : skybox_verts) glVertex3fv(&v[0]);
        glEnd();
    });
}

GLuint pbr_tools::compute_brdf_integration_map() const
{
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
    glDeleteFramebuffers(1, &fbo);

    return brdf_integration_map;
}

void pbr_tools::draw_skybox(GLuint cubemap, const float4x4 & view_proj_matrix) const
{
    glUseProgram(cubemap_skybox_prog);
    glUniformMatrix4fv(glGetUniformLocation(cubemap_skybox_prog, "u_view_proj_matrix"), 1, GL_FALSE, &view_proj_matrix[0][0]);
    glUniform1i(glGetUniformLocation(cubemap_skybox_prog, "u_texture"), 0);
    glBindTextureUnit(0, cubemap);
    glDepthMask(GL_FALSE);
    glBegin(GL_QUADS);
    for(auto & v : skybox_verts) glVertex3fv(&v[0]);
    glEnd();
    glDepthMask(GL_TRUE);
}

