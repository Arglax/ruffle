// -*- glsl -*-
#version 100

#ifdef GL_FRAGMENT_PRECISION_HIGH
    precision highp float;
#else
    precision mediump float;
#endif

// Flash's complex blend modes, as the wgpu renderer's shaders/blend/*.wgsl compute them (premultiplied
// alpha). u_parent is what was drawn behind the object, u_current the object drawn on its own.
uniform sampler2D u_parent;
uniform sampler2D u_current;
uniform int u_mode; // 2 darken, 3 difference, 4 invert, 5 overlay, 6 hard light

varying vec2 frag_uv;

vec3 blend_func(vec3 s, vec3 d) {
    if (u_mode == 2) {
        return min(s, d);
    }
    if (u_mode == 3) {
        return abs(d - s);
    }
    if (u_mode == 5) {
        return mix(1.0 - 2.0 * (1.0 - d) * (1.0 - s), 2.0 * s * d, step(d, vec3(0.5)));
    }
    if (u_mode == 6) {
        return mix(1.0 - 2.0 * (1.0 - d) * (1.0 - s), 2.0 * s * d, step(s, vec3(0.5)));
    }
    return s;
}

void main() {
    vec4 dst = texture2D(u_parent, frag_uv);
    vec4 src = texture2D(u_current, frag_uv);
    if (src.a <= 0.0) {
        // Leave the pixels behind untouched (keeps their anti-aliasing samples).
        discard;
    }
    vec3 d = dst.a > 0.0 ? dst.rgb / dst.a : vec3(0.0);
    vec3 mixed = u_mode == 4 ? (1.0 - d) : blend_func(src.rgb / src.a, d);
    vec3 rgb = src.rgb * (1.0 - dst.a) + dst.rgb * (1.0 - src.a) + src.a * dst.a * mixed;
    gl_FragColor = vec4(rgb, src.a + dst.a * (1.0 - src.a));
}
