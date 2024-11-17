#version 330 core
#define saturate(x) clamp(x, 0.0, 1.0)
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D grid;

vec3 TurboColormap(in float x) {
    const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

    x = saturate(x);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(
        dot(v4, kRedVec4)   + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4)  + dot(v2, kBlueVec2));
}

// https://www.shadertoy.com/view/XsSXDy
vec4 powers(float x) {
    return vec4(x*x*x, x*x, x, 1.0);
}

vec4 ca = vec4(3.0,  -6.0,   0.0,  4.0) /  6.0;
vec4 cb = vec4(-1.0,   6.0, -12.0,  8.0) /  6.0;

vec4 spline(float x, vec4 c0, vec4 c1, vec4 c2, vec4 c3) {
    return c0 * dot(cb, powers(x + 1.0)) + 
    c1 * dot(ca, powers(x)) +
    c2 * dot(ca, powers(1.0 - x)) +
    c3 * dot(cb, powers(2.0 - x));
}

#define SAM(a,b)  texture(grid, (i+vec2(float(a),float(b))+0.5)/res, -99.0)

vec4 texture_Bicubic(sampler2D tex, vec2 t) {
    vec2 res = textureSize(tex, 0);
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline(f.y, spline(f.x, SAM(-1,-1), SAM(0,-1), SAM(1,-1), SAM(2,-1)),
        spline(f.x, SAM(-1, 0), SAM(0, 0), SAM(1, 0), SAM(2, 0)),
        spline(f.x, SAM(-1, 1), SAM(0, 1), SAM(1, 1), SAM(2, 1)),
        spline(f.x, SAM(-1, 2), SAM(0, 2), SAM(1, 2), SAM(2, 2)));
}

void main() {
    vec4 interpolatedValue = texture_Bicubic(grid, TexCoord) / 25.0;
    float value = interpolatedValue.r;

    vec3 color = TurboColormap(value);

    FragColor = vec4(color, 1.0);
}