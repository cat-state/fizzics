const time_step: f32 = 1.0 / 60.0;
const gravity: vec3<f32> = vec3<f32>(0.0, -9.8, 0.0);
const constraint_stiffness: f32 = 0.5;
const rest_length: f32 = 0.5;
const collision_damping: f32 = 0.01;
const mouse_attraction_strength: f32 = 10.0;
const num_cubes: i32 = 4*4*4;
const vertices_per_cube: i32 = 8;
const total_vertices: i32 = num_cubes * vertices_per_cube;
const cube_collision_radius: f32 = 0.33;
const boundary: vec3<f32> = vec3<f32>(4.0, 4.0, 4.0);
struct Particle {
    x: vec3<f32>,
    mass: f32,
    v: vec3<f32>,
    _padding: f32,
}

struct Voxel {
    particles: array<Particle, 8>,
}

struct Uniforms {
    i_mouse: vec4<f32>,
    i_resolution: vec2<f32>,
    i_frame: i32,
}

fn get_voxel(idx: u32) -> Voxel {
    return Voxel(
        array<Particle, 8>(
            particles[idx * 8 + 0],
            particles[idx * 8 + 1],
            particles[idx * 8 + 2],
            particles[idx * 8 + 3],
            particles[idx * 8 + 4],
            particles[idx * 8 + 5],
            particles[idx * 8 + 6],
            particles[idx * 8 + 7],
        )
    );
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;


fn project(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return dot(a, b) / dot(b, b) * b;
}

fn gram_schmidt(A: vec3<f32>, B: vec3<f32>, C: vec3<f32>) -> array<vec3<f32>, 3> {
    var Ao = normalize(A);
    var Bo = B - project(B, Ao);
    Bo = normalize(Bo);
    var Co = C - project(C, Ao) - project(C, Bo);
    Co = normalize(Co);
    return array<vec3<f32>, 3>(Ao, Bo, Co);
}

fn slerp(start: vec3<f32>, end: vec3<f32>, t: f32) -> vec3<f32> {
    var cos_theta = dot(start, end);
    
    if (cos_theta > 0.9995) {
        return normalize(mix(start, end, t));
    }
    
    cos_theta = clamp(cos_theta, -1.0, 1.0);
    let theta = acos(cos_theta);
    let sin_theta = sin(theta);
    
    let start_weight = sin((1.0 - t) * theta) / sin_theta;
    let end_weight = sin(t * theta) / sin_theta;
    
    return normalize(start_weight * start + end_weight * end);
}

fn average_on_sphere(v1: vec3<f32>, v2: vec3<f32>, v3: vec3<f32>) -> vec3<f32> {
    let v12 = slerp(v1, v2, 0.5);
    return slerp(v12, v3, 1.0 / 3.0);
}

fn apply_gram_schmidt_constraint(_voxel: Voxel) -> Voxel {
    var voxel = _voxel;
    for (var i = 0; i < vertices_per_cube; i++) {
        let next1 = (i + 1) % 4 + (i / 4) * 4;
        let next2 = (i + 3) % 4 + (i / 4) * 4;
        let next3 = i ^ 4;

        let edge1 = voxel.particles[next1].x - voxel.particles[i].x;
        let edge2 = voxel.particles[next2].x - voxel.particles[i].x;
        let edge3 = voxel.particles[next3].x - voxel.particles[i].x;

        let gs_0 = gram_schmidt(edge1, edge2, edge3);
        let gs_1 = gram_schmidt(edge2, edge3, edge1);
        let gs_2 = gram_schmidt(edge3, edge1, edge2);

        // u1 = average_on_sphere(u1, gs[1], gs[2]);
        // u2 = average_on_sphere(u2, gs[2], gs[0]);
        // u3 = average_on_sphere(u3, gs[0], gs[1]);

        var u1 = normalize(gs_0[0] + gs_2[1] + gs_1[2]);
        var u2 = normalize(gs_0[1] + gs_1[0] + gs_2[2]);
        var u3 = normalize(gs_0[2] + gs_1[1] + gs_2[0]);
        u1 *= rest_length;
        u2 *= rest_length;
        u3 *= rest_length;

        let ideal_next1 = voxel.particles[i].x + u1;
        let ideal_next2 = voxel.particles[i].x + u2;
        let ideal_next3 = voxel.particles[i].x + u3;

        let correction_next1 = (ideal_next1 - voxel.particles[next1].x) * constraint_stiffness;
        let correction_next2 = (ideal_next2 - voxel.particles[next2].x) * constraint_stiffness;
        let correction_next3 = (ideal_next3 - voxel.particles[next3].x) * constraint_stiffness;

        // this has no transational bias but causes voxels to spin 
        // let total_correction = correction_next1 + correction_next2 + correction_next3;
        // let correction_self = -total_correction / 4.0;

        // voxel.particles[next1].x += correction_next1 * 0.25;
        // voxel.particles[next2].x += correction_next2 * 0.25;
        // voxel.particles[next3].x += correction_next3 * 0.25;
        // voxel.particles[i].x += correction_self;

        let correction_self = -(correction_next1 + correction_next2 + correction_next3) / 3.0;

        voxel.particles[next1].x += correction_next1;
        voxel.particles[next2].x += correction_next2;
        voxel.particles[next3].x += correction_next3;
        voxel.particles[i].x += correction_self;
    }
    return voxel;
}

fn handle_boundary_collisions(_particle: Particle) -> Particle {
    var particle = _particle;
    if (particle.x.x < -boundary.x) {
        particle.x.x = -boundary.x;
        particle.v.x *= -collision_damping;
    } else if (particle.x.x > boundary.x) {
        particle.x.x = boundary.x;
        particle.v.x *= -collision_damping;
    }

    if (particle.x.y < -boundary.y) {
        particle.x.y = -boundary.y;
        particle.v.y *= -collision_damping;
    } else if (particle.x.y > boundary.y) {
        particle.x.y = boundary.y;
        particle.v.y *= -collision_damping;
    }

    if (particle.x.z < -boundary.z) {
        particle.x.z = -boundary.z;
        particle.v.z *= -collision_damping;
    } else if (particle.x.z > boundary.z) {
        particle.x.z = boundary.z;
        particle.v.z *= -collision_damping;
    }
    return particle;
}

@compute @workgroup_size(1, 1, 1)
fn handle_particle_collisions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    var p_i = particles[particle_index];
    let total_particles = arrayLength(&particles);
    for (var j = particle_index + 1u; j <total_particles; j++) {
        if (particle_index / u32(vertices_per_cube) == j / u32(vertices_per_cube)) {
            continue;
        }

        var p_j = particles[j];
        let diff = p_i.x - p_j.x;
        let distance = length(diff);
        
        if (distance < 2.0 * cube_collision_radius) {
            let collision_normal = normalize(diff);
            let overlap = 2.0 * cube_collision_radius - distance;
            
            let v1 = p_i.v;
            let v2 = p_j.v;
            
            let relative_velocity = v1 - v2;
            let velocity_along_normal = dot(relative_velocity, collision_normal);
            
            if (velocity_along_normal > 0.0) {
                continue;
            }
            
            let restitution = 0.5;
            var jr = -(1.0 + restitution) * velocity_along_normal;
            jr /= 2.0;
            
            let impulse = jr * collision_normal;
            
            p_i.x += (impulse + overlap * 0.5 * collision_normal) * (1.0 - collision_damping);
            p_j.x -= (impulse + overlap * 0.5 * collision_normal) * (1.0 - collision_damping);
            // data race
            particles[j] = p_j;
        }
    }
    // also data race
    particles[particle_index] = p_i;
}

fn find_closest_vertex(mouse_pos: vec3<f32>, voxel: ptr<function, Voxel>) -> i32 {
    var min_dist = distance(mouse_pos, (*voxel).particles[0].x);
    var closest_index = -1;
    for (var j = 0; j < vertices_per_cube; j++) {
        let dist = distance(mouse_pos, (*voxel).particles[j].x);
        if (dist < min_dist) {
            min_dist = dist;
            closest_index = j;
        }
    }
    return closest_index;
}

@compute @workgroup_size(1, 1, 1)
fn voxel_constraint(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cube_index = global_id.x;
    
    var voxel = get_voxel(cube_index);
    let mouse_pos = vec3<f32>((uniforms.i_mouse.xy - 0.5 * uniforms.i_resolution) / uniforms.i_resolution.y, 0.0);

    var closest_index = -1;
    if (uniforms.i_mouse.z > 0.0) {
        //closest_index = find_closest_vertex(mouse_pos, &voxel);
    }

    for (var i = 0; i < vertices_per_cube; i++) {
        var particle = voxel.particles[i];
        var curr_pos = particle.x;
        var velocity = particle.v;
        var new_pos = curr_pos + velocity * time_step + gravity * time_step * time_step;
        particle.x = new_pos;
        particle = handle_boundary_collisions(particle);
        voxel.particles[i] = particle;
        // if (i == closest_index) {
        //     let to_mouse = mouse_pos - new_pos;
        //     new_pos += to_mouse * mouse_attraction_strength * time_step;
        // }
    }

    voxel = apply_gram_schmidt_constraint(voxel);
    
    for (var i = 0; i < vertices_per_cube; i++) {
        let global_particle_index = cube_index * u32(vertices_per_cube) + u32(i);
        var particle = voxel.particles[i];
        var new_pos = particle.x;
        var velocity = particle.v;
        particle = handle_boundary_collisions(particle);
        particle.x = new_pos;
        particle.v = (new_pos - particle.x) / time_step;
        particles[global_particle_index] = particle;
        // particles[particle_index].x.y += time_step;
    }
}