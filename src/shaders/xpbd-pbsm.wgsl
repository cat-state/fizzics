// Physically Based Shape Matching

struct Particle {
    x: vec3<f32>,
    mass: f32,
    v: vec3<f32>,
    _padding: f32,
    x_prev: vec3<f32>,
    _padding2: f32,
}

struct Voxel {
    particles: array<Particle, 8>,
}



@compute
fn compute() {

}