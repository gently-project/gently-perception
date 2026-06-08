#version 330 core

// Pixel-equivalent port of the raymarcher in
// gently-annotator/static/js/viewer.js (FRAGMENT_SHADER, lifted from gently
// origin/main commit cf15449). Only the desktop-GL boilerplate has been
// touched: no precision qualifiers, GLSL 3.30 instead of 3.00 ES, and `step`
// renamed to `stepVec` since `step` is a builtin in some strict drivers.

uniform sampler3D uVolume;
uniform vec3 uBoxSize;
uniform float uThreshold;
uniform float uContrast;
uniform vec3 uCameraObjectPos;
uniform int uMaxSteps;

in vec3 vObjectPos;
out vec4 outColor;

bool rayBoxIntersect(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax,
                     out float tMin, out float tMax) {
    vec3 invD = 1.0 / rd;
    vec3 t1 = (boxMin - ro) * invD;
    vec3 t2 = (boxMax - ro) * invD;
    vec3 tmn = min(t1, t2);
    vec3 tmx = max(t1, t2);
    tMin = max(max(tmn.x, tmn.y), tmn.z);
    tMax = min(min(tmx.x, tmx.y), tmx.z);
    return tMax > max(tMin, 0.0);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec3 boxHalf = uBoxSize * 0.5;
    vec3 ro = uCameraObjectPos;
    vec3 rd = normalize(vObjectPos - uCameraObjectPos);

    float tMin, tMax;
    if (!rayBoxIntersect(ro, rd, -boxHalf, boxHalf, tMin, tMax)) discard;
    tMin = max(tMin, 0.0);

    float totalLen = tMax - tMin;
    float stepSize = totalLen / float(uMaxSteps);
    float jitter = hash12(gl_FragCoord.xy) * stepSize;
    vec3 pos = ro + rd * (tMin + jitter);
    vec3 stepVec = rd * stepSize;

    const float NOMINAL_STEPS = 192.0;
    float opacityScale = NOMINAL_STEPS / float(uMaxSteps);

    vec4 accum = vec4(0.0);
    for (int i = 0; i < 512; i++) {
        if (i >= uMaxSteps) break;
        vec3 uvw = (pos + boxHalf) / uBoxSize;
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
            pos += stepVec;
            continue;
        }
        float sampleVal = texture(uVolume, uvw).r;
        float density = smoothstep(uThreshold, min(uThreshold + 0.45, 1.0), sampleVal);
        if (density > 0.001) {
            float v = clamp((sampleVal - 0.5) * uContrast + 0.5, 0.0, 1.0);
            vec3 color = vec3(v);
            float alpha = density * 0.18 * opacityScale;
            accum.rgb += (1.0 - accum.a) * color * alpha;
            accum.a += (1.0 - accum.a) * alpha;
        }
        pos += stepVec;
        if (accum.a > 0.999) break;
    }
    if (accum.a < 0.005) discard;
    outColor = accum;
}
