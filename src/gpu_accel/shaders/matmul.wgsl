@group(0) @binding(0) var<storage, read> lhs: arr<f32>;
@group(0) @binding(1) var<storage, read> rhs: arr<f32>;
@group(0) @binding(2) var<uniform> dims: vec4u;
@group(0) @binding(3) var<storage, read_write> output: arr<f32>;

@compute
@workgroup_size(1)
fn matmul(@builtin(global_invocation_id) pos: vec3u) {
    output[pos.x * dims.x + pos.y] = 0;
    for (var i: u32 = 0; i < dims.w; i++) {
        output[pos.x * dims.x + pos.y] += lhs[pos.x * dims.x + i] * rhs[i * dims.z + pos.y];
    }
}
