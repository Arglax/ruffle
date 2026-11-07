#version 100

// A full-target quad for mixing a blend layer with what is behind it (position 0..1 = uv).
attribute vec2 position;
uniform vec2 u_uv_scale; // the part of the (pooled, larger) textures in use

varying vec2 frag_uv;

void main() {
    frag_uv = position * u_uv_scale;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
