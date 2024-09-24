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

struct Force {
    f: vec3<f32>,
    particle_index: i32,
}

struct FaceConstraint {
    particle_indices: array<u32, 8>,
}

struct ContactConstraint {
    num_contacts: atomic<u32>,
    particle_indices: array<vec2<u32>>,
}

struct Uniforms {
    h: f32,
    boundary_min: vec3<f32>,
    boundary_max: vec3<f32>,
    particle_radius: f32,
}

struct NeoHookean {
    lambda: f32,
    mu: f32,
    gamma: f32,
}

// Add this struct to store F matrices
struct FMatrices {
    F_h: mat3x3<f32>,
    F_d: mat3x3<f32>,
    lagrange_h: f32,
    lagrange_d: f32,
    grad_C_h: array<vec3<f32>, 8>,
    grad_C_d: array<vec3<f32>, 8>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forces: array<Force>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read_write> f_matrices: array<FMatrices>;
@group(0) @binding(4) var<storage, read_write> face_constraints: array<FaceConstraint>;

@compute @workgroup_size(1, 1, 1)
fn apply_velocity_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    var particle = particles[particle_index];
    particle.x_prev = particle.x;
    let num_forces = 1;
    for (var i = 0; i < num_forces; i++) {
        let force = forces[i];
        if (force.particle_index < 0 || force.particle_index == i32(particle_index)) {
            particle.v += uniforms.h * force.f / particle.mass;
        }
    }
    particle.x += particle.v * uniforms.h;
    particles[particle_index] = particle;
}

fn outer_product(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return (mat3x3<f32>(a.x * b.x, a.y * b.x, a.z * b.x,
                        a.x * b.y, a.y * b.y, a.z * b.y,
                        a.x * b.z, a.y * b.z, a.z * b.z));
}

fn trace(m: mat3x3<f32>) -> f32 {
    return m[0][0] + m[1][1] + m[2][2];
}

fn shape_matching(_voxel: array<Particle, 8>, _rest_positions: array<vec3<f32>, 8>, volume: f32, q_inv: mat3x3<f32>, material: NeoHookean, voxel_index: u32)
-> array<Particle, 8> {
    var voxel = _voxel;
    var rest_positions = _rest_positions;
    var com = vec3<f32>(0.0);
    var total_mass = 0.0;
    for (var i = 0; i < 8; i++) {
        com += voxel[i].x * voxel[i].mass;
        total_mass += voxel[i].mass;
    }
    com = com * (1.0 / total_mass);
    // Recompute F after C_h constraint

    var F = mat3x3<f32>(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = F * q_inv;
    // Solve C_d constraint
    let C_d = sqrt(trace(transpose(F) * F));
    let r = length(vec3(length(F[0]), length(F[1]), length(F[2])));
    var grad_C_d = array<vec3<f32>, 8>();
    for (var i = 0; i < 8; i++) {
        grad_C_d[i] = (voxel[i].mass / r) * F * transpose(q_inv) * rest_positions[i];
    }

    var sum_grad_norms_d = 0.0;
    for (var i = 0; i < 8; i++) {
        let norm = length(grad_C_d[i]);
        sum_grad_norms_d += (norm * norm) / voxel[i].mass;
    }

    let alpha_d = 1.0 / (volume * material.mu);
    let largange_multiplier_d = -C_d / (sum_grad_norms_d + (alpha_d / (uniforms.h * uniforms.h)));

    // After calculating largange_multiplier_d
    f_matrices[voxel_index].lagrange_d = largange_multiplier_d;

    for (var i = 0; i < 8; i++) {
        voxel[i].x += largange_multiplier_d * grad_C_d[i] / voxel[i].mass;
    }

    // After C_d constraint
    f_matrices[voxel_index].F_d = F;

    // After calculating grad_C_d
    for(var i = 0; i < 8; i++) {
        grad_C_d[i] *= largange_multiplier_d;
    }
    f_matrices[voxel_index].grad_C_d = grad_C_d;


    com = vec3<f32>(0.0);
    total_mass = 0.0;
    for (var i = 0; i < 8; i++) {
        com += voxel[i].x * voxel[i].mass;
        total_mass += voxel[i].mass;
    }
    com = com * (1.0 / total_mass);

    F = mat3x3<f32>(0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = F * q_inv;


    // Solve C_h constraint
    var C_h = determinant(F) - material.gamma;
    let ch_fs = mat3x3<f32>(
        cross(F[1], F[2]),
        cross(F[2], F[0]),
        cross(F[0], F[1])
    ) * transpose(q_inv);

    var grad_C_h = array<vec3<f32>, 8>();
    for (var i = 0; i < 8; i++) {
        grad_C_h[i] = voxel[i].mass * ch_fs * rest_positions[i];
    }

    var sum_grad_norms_h = 0.0;
    for (var i = 0; i < 8; i++) {
        let norm = length(grad_C_h[i]);
        sum_grad_norms_h += (norm * norm) / voxel[i].mass;
    }

    let alpha_h = 1.0 / (volume * material.lambda);
    let largange_multiplier_h = -C_h / (sum_grad_norms_h + (alpha_h / (uniforms.h * uniforms.h)));

    // After calculating largange_multiplier_h
    f_matrices[voxel_index].lagrange_h = largange_multiplier_h;

    for (var i = 0; i < 8; i++) {
        voxel[i].x += largange_multiplier_h * grad_C_h[i] / voxel[i].mass;
    }

    // After C_h constraint
    f_matrices[voxel_index].F_h = F;

    // After calculating grad_C_h
    for(var i = 0; i < 8; i++) {
        grad_C_h[i] *= largange_multiplier_h;
    }
    f_matrices[voxel_index].grad_C_h = grad_C_h;

    
    com = vec3<f32>(0.0);
    total_mass = 0.0;
    for (var i = 0; i < 8; i++) {
        com += voxel[i].x * voxel[i].mass;
        total_mass += voxel[i].mass;
    }
    com = com * (1.0 / total_mass);

    F = mat3x3<f32>(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = F * q_inv;

    for (var i = 0; i < 8; i++) {
        voxel[i].x = (F * rest_positions[i]) + com;
    }

    return voxel;
}
@compute @workgroup_size(1, 1, 1)
fn cube_voxel_shape_matching(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_index = global_id.x;

    var voxel = array<Particle, 8>(
        particles[voxel_index * 8],
        particles[voxel_index * 8 + 1],
        particles[voxel_index * 8 + 2],
        particles[voxel_index * 8 + 3],
        particles[voxel_index * 8 + 4],
        particles[voxel_index * 8 + 5],
        particles[voxel_index * 8 + 6],
        particles[voxel_index * 8 + 7]
    );
    let rest_length = 2.0;
    let h_length = rest_length / 2.0;
    var rest_positions = array<vec3<f32>, 8>(
        vec3(-1.0, -1.0, -1.0) * h_length, // 0b000
        vec3(1.0, -1.0, -1.0) * h_length, // 0b001
        vec3(1.0, 1.0, -1.0) * h_length, // 0b011
        vec3(-1.0, 1.0, -1.0) * h_length, // 0b010
        vec3(-1.0, -1.0, 1.0) * h_length, // 0b100
        vec3(1.0, -1.0, 1.0) * h_length, // 0b101
        vec3(1.0, 1.0, 1.0) * h_length, // 0b111
        vec3(-1.0, 1.0, 1.0) * h_length, // 0b110
    );

    // for a cube, its Q is diag(vec3(volume))
    // so inv(Q) is diag(vec3(1.0 / volume))
    let volume = rest_length * rest_length * rest_length;
    var Q = mat3x3<f32>(0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        Q += outer_product(rest_positions[i], rest_positions[i]);
    }
    let q_inv = invert(Q);

    let E = 0.1; // youngs modulus of rubber
    let nu = 0.4; // poissons ratio of rubber
    let mu = E / (2.0 * (1.0 + nu));
    let lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let material = NeoHookean(lambda, mu, 1.0 + (mu / lambda));

    voxel = shape_matching(voxel, rest_positions, volume, q_inv, material, voxel_index);
    for (var i: u32 = 0; i < 8; i++) {
        particles[voxel_index * 8 + i] = voxel[i];
    }
}

fn apply_face_constraint(_face_constraint: FaceConstraint) {    
    var face_constraint = _face_constraint;
    // Load particles referred by the face constraint
    var face_particles: array<Particle, 8>;
    for (var i = 0u; i < 8u; i++) {
        face_particles[i] = particles[face_constraint.particle_indices[i]];
    }

    // Treat these particles as a voxel and run shape matching
    let rest_length = 2.0;
    let h_length = rest_length / 2.0;
    var rest_positions = array<vec3<f32>, 8>(
        vec3(-1.0, -1.0, -1.0) * h_length, // 0b000
        vec3(1.0, -1.0, -1.0) * h_length, // 0b001
        vec3(1.0, 1.0, -1.0) * h_length, // 0b011
        vec3(-1.0, 1.0, -1.0) * h_length, // 0b010
        vec3(-1.0, -1.0, 1.0) * h_length, // 0b100
        vec3(1.0, -1.0, 1.0) * h_length, // 0b101
        vec3(1.0, 1.0, 1.0) * h_length, // 0b111
        vec3(-1.0, 1.0, 1.0) * h_length, // 0b110
    );


    let volume = rest_length * rest_length * rest_length;
    var Q = mat3x3<f32>(0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        Q += outer_product(rest_positions[i], rest_positions[i]);
    }
    let q_inv = invert(Q);

    let E = 2.3e1; // Young's modulus of rubber
    let nu = 0.4; // Poisson's ratio of rubber
    let mu = E / (2.0 * (1.0 + nu));
    let lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let material = NeoHookean(lambda, mu, 1.0 + (mu / lambda));

    face_particles = shape_matching(face_particles, rest_positions, volume, q_inv, material, u32(0));

    // Update the original particles with the shape-matched results
    for (var i = 0u; i < 8u; i++) {
        particles[face_constraint.particle_indices[i]] = face_particles[i];
    }
}

@compute @workgroup_size(1, 1, 1)
fn apply_x_face_constraint(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let face_constraint = face_constraints[global_id.x];
    apply_face_constraint(face_constraint);
}

@compute @workgroup_size(1, 1, 1)
fn apply_y_face_constraint(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_constraints = num_workgroups.x;
    let face_constraint = face_constraints[num_constraints + global_id.x];
    apply_face_constraint(face_constraint);
}

@compute @workgroup_size(1, 1, 1)
fn apply_z_face_constraint(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_constraints = num_workgroups.x;
    let face_constraint = face_constraints[num_constraints * 2 + global_id.x];
    apply_face_constraint(face_constraint);
}

@compute @workgroup_size(1, 1, 1)
fn boundary_constraints(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    var particle = particles[particle_index];
    let damping = 0.9999; // Slight damping factor
    let epsilon = 0.0; //0.001; // Small offset to prevent sticking to the boundary

    // Check each dimension
    for (var dim = 0; dim < 3; dim++) {
        if (particle.x[dim] < uniforms.boundary_min[dim]) {
            // Particle is outside the minimum boundary
            let t = (uniforms.boundary_min[dim] - particle.x_prev[dim]) / (particle.x[dim] - particle.x_prev[dim]);
            particle.x[dim] = mix(particle.x_prev[dim], particle.x[dim], t) + epsilon;
            particle.v[dim] = -particle.v[dim] * damping;
        } else if (particle.x[dim] > uniforms.boundary_max[dim]) {
            // Particle is outside the maximum boundary
            let t = (uniforms.boundary_max[dim] - particle.x_prev[dim]) / (particle.x[dim] - particle.x_prev[dim]);
            particle.x[dim] = mix(particle.x_prev[dim], particle.x[dim], t) - epsilon;
            particle.v[dim] = -particle.v[dim] * damping;
        }
    }
    particles[particle_index] = particle;
}

@compute @workgroup_size(1, 1, 1)
fn update_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    var particle = particles[particle_index];
    particle.v = (particle.x - particle.x_prev) / uniforms.h;
    particles[particle_index] = particle;
}

fn invert(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;

    return mat3x3<f32>(
        (e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det
    );
}


@compute @workgroup_size(1, 1, 1)
fn apply_damping(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_index = global_id.x;
    var voxel = array<Particle, 8>(
        particles[voxel_index * 8],
        particles[voxel_index * 8 + 1],
        particles[voxel_index * 8 + 2],
        particles[voxel_index * 8 + 3],
        particles[voxel_index * 8 + 4],
        particles[voxel_index * 8 + 5],
        particles[voxel_index * 8 + 6],
        particles[voxel_index * 8 + 7]
    );

    // Calculate center of mass velocity
    var com_velocity = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0; i < 8; i++) {
        com_velocity += voxel[i].v;
    }
    com_velocity /= 8.0;

    // Calculate angular velocity
    var angular_momentum = vec3<f32>(0.0, 0.0, 0.0);
    var inertia_tensor = mat3x3<f32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var com_position = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0; i < 8; i++) {
        com_position += voxel[i].x;
    }
    com_position /= 8.0;

    for (var i: u32 = 0; i < 8; i++) {
        let r = voxel[i].x - com_position;
        angular_momentum += cross(r, voxel[i].v - com_velocity);
        inertia_tensor += mat3x3<f32>(
            r.y * r.y + r.z * r.z, -r.x * r.y, -r.x * r.z,
            -r.y * r.x, r.x * r.x + r.z * r.z, -r.y * r.z,
            -r.z * r.x, -r.z * r.y, r.x * r.x + r.y * r.y
        );
    }
    let angular_velocity = invert(inertia_tensor) * angular_momentum;

    // Apply damping
    let damping_factor = 0.00; // Adjust as needed
    for (var i: u32 = 0; i < 8; i++) {
        let r = voxel[i].x - com_position;
        let damped_velocity = com_velocity + cross(angular_velocity, r);
        voxel[i].v = damping_factor * damped_velocity + (1.0 - damping_factor) * voxel[i].v;
        particles[voxel_index * 8 + i] = voxel[i];
    }
}
