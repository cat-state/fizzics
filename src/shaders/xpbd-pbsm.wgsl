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

struct ConstraintPartition {
    start: u32,
    end: u32,
}

struct FaceConstraint {
    particle_indices: array<u32, 8>,
}

struct LagrangeMultipliers {
    lagrange_h: f32,
    lagrange_d: f32,
}

struct ContactConstraint {
    num_contacts: atomic<u32>,
    particle_indices: array<vec2<u32>>,
}

struct Uniforms {
    h: f32,
    num_particles: u32,
    num_voxels: u32,
    num_constraint_partitions: u32,
    boundary_min: vec3<f32>,
    s: f32,
    boundary_max: vec3<f32>,
    particle_radius: f32,
    rest_length: f32, 
    inv_q: mat3x3<f32>,
}

struct NeoHookean {
    lambda: f32,
    mu: f32,
    gamma: f32,
}

struct Hookean {
    E: f32,
    nu: f32,
}

// Add this struct to store F matrices
struct FMatrices {
    F_h: mat3x3<f32>,
    F_d: mat3x3<f32>,
    lagrange_h: f32,
    lagrange_d: f32,
    C_h: f32,
    C_d: f32,
    grad_C_h: array<vec3<f32>, 8>,
    grad_C_d: array<vec3<f32>, 8>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> forces: array<Force>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read_write> f_matrices: array<FMatrices>;
@group(0) @binding(4) var<storage, read_write> face_constraints: array<FaceConstraint>;
@group(0) @binding(5) var<storage, read_write> constraint_partitions: array<ConstraintPartition>;

@group(0) @binding(6) var<storage, read_write> current_partition: u32;

@group(0) @binding(7) var<storage, read_write> lagrange_multipliers: array<LagrangeMultipliers>;

// @compute @workgroup_size(256, 1, 1)
// fn apply_velocity_forces(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
//     if (global_id.x >= uniforms.num_particles) {
//         return;
//     }
//     let particle_index = global_id.x;
//     var particle = particles[particle_index];
//     particle.x_prev = particle.x;
//     let num_forces = 1;
//     for (var i = 0; i < num_forces; i++) {
//         let force = forces[i];
//         if (force.particle_index < 0 || force.particle_index == i32(particle_index)) {
//             particle.v += uniforms.h * force.f / particle.mass;
//         }
//     }
//     particle.x += particle.v * uniforms.h;
//     particles[particle_index] = particle;
// }

@compute @workgroup_size(256, 1, 1)
fn apply_velocity_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= uniforms.num_particles) {
        return;
    }
    let particle_index = global_id.x;
    var particle = particles[particle_index];
    particle.x_prev = particle.x;

    // Sum all forces acting on the particle
    var total_force = vec3<f32>(0.0, 0.0, 0.0);
    let num_forces = 1;
    for (var i = 0; i < num_forces; i++) {
        let force = forces[i];
        if (force.particle_index < 0 || force.particle_index == i32(particle_index)) {
            total_force += force.f;
        }
    }
    particle.v += uniforms.h * total_force / particle.mass;
    // Update position using explicit integration (Position Verlet)
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

// fn stable_com(_particles: array<Particle, 8>) -> vec3<f32> {
//     var particles = _particles;
//     var weighted_mean = vec3<f32>(0.0);
//     var total_mass = 0.0;
//     for (var i = 0u; i < 8u; i++) {
//         total_mass += particles[i].mass;
//         weighted_mean += (particles[i].mass * (particles[i].x - weighted_mean)) / total_mass;
//     }
//     return weighted_mean;
// }

fn stable_com(_particles: array<Particle, 8>) -> vec3<f32> {
    var particles = _particles;
    var weighted_sum = vec3<f32>(0.0);
    var total_mass = 0.0;
    for (var i = 0u; i < 8u; i++) {
        total_mass += particles[i].mass;
        weighted_sum += particles[i].mass * particles[i].x;
    }
    return weighted_sum / total_mass;
}

fn solve_2x2_system(A: mat2x2<f32>, b: vec2<f32>) -> vec2<f32> {
    var local_A = A;
    var local_b = b;
    var x = vec2<f32>(0.0, 0.0);
    var p = vec2<u32>(0u, 1u);

    // Pivoting
    if abs(local_A[1][0]) > abs(local_A[0][0]) {
        // Swap rows
        let temp_row = local_A[0];
        local_A[0] = local_A[1];
        local_A[1] = temp_row;
        let temp_b = local_b.x;
        local_b.x = local_b.y;
        local_b.y = temp_b;
        p = vec2<u32>(1u, 0u);
    }

    // Check for singularity
    if local_A[0][0] == 0.0 {
        // Matrix is singular, return zero vector
        return vec2<f32>(0.0, 0.0);
    }

    // LU decomposition
    let L21 = local_A[1][0] / local_A[0][0];
    local_A[1][0] = L21;
    local_A[1][1] = local_A[1][1] - L21 * local_A[0][1];

    // Forward substitution
    x.x = local_b.x;
    x.y = local_b.y - L21 * x.x;

    // Backward substitution
    if local_A[1][1] == 0.0 {
        // Matrix is singular, return zero vector
        return vec2<f32>(0.0, 0.0);
    }
    x.y = x.y / local_A[1][1];
    x.x = (x.x - local_A[0][1] * x.y) / local_A[0][0];

    // Unpivot the solution
    return vec2<f32>(x[p[0]], x[p[1]]);
}


struct TetShapeMatchingResult {
    voxel: array<Particle, 4>,
    lagrange_multiplier: LagrangeMultipliers,
}

struct ShapeMatchingResult {
    voxel: array<Particle, 8>,
    lagrange_multiplier: LagrangeMultipliers,
}

fn decoupled_hookean_shape_matching(_voxel: array<Particle, 8>, 
                   _rest_positions: array<vec3<f32>, 8>, 
                   volume: f32, 
                   q_inv: mat3x3<f32>, 
                   material: Hookean, 
                   voxel_index: u32,
                   _lagrange_multiplier: LagrangeMultipliers,
                   )
-> ShapeMatchingResult {
    var voxel = _voxel;
    var rest_positions = _rest_positions;
    var lagrange_multiplier = _lagrange_multiplier;
    var com = stable_com(voxel); 

    var F = mat3x3<f32>(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = uniforms.s * F * q_inv;

    let identity = mat3x3<f32>(1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0);
    let strain = 0.5 * (transpose(F) * F - identity);

    let C_scale = 1.0 / ((1.0 + material.nu) * (1.0 - 2.0 * material.nu));
    let nu = material.nu;
    let C_ii = mat3x3<f32>(
        1.0 - nu, nu, nu,
        nu, 1.0 - nu, nu,
        nu, nu, 1.0 - nu
    );
    let C_ij = mat3x3<f32>(
        1.0 - 2.0 * material.nu, 0.0, 0.0,
        0.0, 1.0 - 2.0 * material.nu, 0.0,
        0.0, 0.0, 1.0 - 2.0 * material.nu
    );
    let stress_ii = (C_scale * C_ii * vec3<f32>(strain[0][0], strain[1][1], strain[2][2]));
    let stress_ij = (C_scale * C_ij * vec3<f32>(strain[0][1], strain[1][2], strain[2][0]));
    let W = 0.5 * (dot(vec3<f32>(strain[0][0], strain[1][1], strain[2][2]), stress_ii) 
                 + dot(vec3<f32>(strain[0][1], strain[1][2], strain[2][0]), stress_ij) * 2.0);
    let C_hooke = sqrt(2.0 * W);
    let alpha = 1.0 / (volume * material.E);
    let S = mat3x3<f32>(
        stress_ii[0], stress_ij[0], stress_ij[2],
        stress_ij[0], stress_ii[1], stress_ij[1],
        stress_ij[2], stress_ij[1], stress_ii[2],
    );
    var grad_C_hooke = array<vec3<f32>, 8>();
    var sum_grad_norms = 0.0;
    for (var i = 0; i < 8; i++) {
        grad_C_hooke[i] =  uniforms.s * (voxel[i].mass / C_hooke) * S * F * q_inv * rest_positions[i];
        sum_grad_norms += dot(grad_C_hooke[i], grad_C_hooke[i]) / voxel[i].mass;
    }

    let lambda = -C_hooke / (sum_grad_norms + (alpha / (uniforms.h * uniforms.h)));

    for (var i = 0; i < 8; i++) {
        voxel[i].x += lambda * grad_C_hooke[i] / voxel[i].mass;
    }
    
    com = stable_com(voxel);
    F = mat3x3<f32>(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = uniforms.s * F * q_inv;

    for (var i = 0; i < 8; i++) {
        voxel[i].x = (F * rest_positions[i]) + com;
    }

    return ShapeMatchingResult(voxel, lagrange_multiplier);
}

    

fn coupled_tetrahedra_shape_matching(_voxel: array<Particle, 4>, 
                   _rest_positions: array<vec3<f32>, 4>, 
                   volume: f32, 
                   q_inv: mat3x3<f32>, 
                   material: NeoHookean, 
                   voxel_index: u32,
                   _lagrange_multiplier: LagrangeMultipliers,
                   )
-> TetShapeMatchingResult {
    var voxel = _voxel;
    var rest_positions = _rest_positions;
    var lagrange_multiplier = _lagrange_multiplier;
    var com = vec3<f32>(0.0, 0.0, 0.0);
    var total_mass = 0.0;
    for (var i = 0; i < 4; i++) {
        com += voxel[i].x * voxel[i].mass;
        total_mass += voxel[i].mass;
    }
    com = com / total_mass;
    var F = mat3x3<f32>(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
    for (var i = 0; i < 4; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = F * q_inv;

    var C_h = determinant(F) - 1.0;//material.gamma;
    var C_d = sqrt(trace(transpose(F) * F)) - sqrt(3.0);
    // Compute gradients
    let ch_fs = mat3x3<f32>(
        cross(F[1], F[2]),
        cross(F[2], F[0]),
        cross(F[0], F[1])
    ) * transpose(q_inv);

    var grad_C_h = array<vec3<f32>, 4>();
    var grad_C_d = array<vec3<f32>, 4>();
    let r = sqrt(dot(F[0], F[0]) + dot(F[1], F[1]) + dot(F[2], F[2]));
    for (var i = 0; i < 4; i++) {
        grad_C_h[i] = voxel[i].mass * ch_fs * rest_positions[i];
        grad_C_d[i] = (voxel[i].mass / r) * F * transpose(q_inv) * rest_positions[i];
    }

    // Compute A matrix
    let alpha_h =  1.0 / (volume * material.lambda);
    let alpha_d =  1.0 / (volume * material.mu);
    var A = mat2x2<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < 4; i++) {
        A[0][0] += dot(grad_C_h[i] / voxel[i].mass, grad_C_h[i]);
        A[0][1] += dot(grad_C_h[i] / voxel[i].mass, grad_C_d[i]);
        A[1][0] += dot(grad_C_d[i] / voxel[i].mass, grad_C_h[i]);
        A[1][1] += dot(grad_C_d[i] / voxel[i].mass, grad_C_d[i]);
    }
    let alpha = vec2<f32>(alpha_h, alpha_d) / (uniforms.h * uniforms.h);
    A[0][0] += alpha[0];
    A[1][1] += alpha[1];

    // Compute b vector
    let b = vec2<f32>(-C_h, -C_d)
          - vec2<f32>(lagrange_multiplier.lagrange_h * alpha[0], 
                      lagrange_multiplier.lagrange_d * alpha[1]);

    // Solve 2x2 linear system
    var x = solve_2x2_system(A, b);

    if(abs(x[0]) < 1e-6) {
        x[0] = 0.0;
    }
    if(abs(x[1]) < 1e-6) {
        x[1] = 0.0;
    }

    let delta_lambda = x;
    lagrange_multiplier.lagrange_h += delta_lambda[0];
    lagrange_multiplier.lagrange_d += delta_lambda[1];

    for (var i = 0; i < 4; i++) {
        // voxel[i].x += (lagrange_multiplier.lagrange_h * grad_C_h[i] + lagrange_multiplier.lagrange_d * grad_C_d[i]) / voxel[i].mass;
    }


    // Store results for visualization or debugging
    return TetShapeMatchingResult(voxel, lagrange_multiplier);
}
    


fn coupled_neohookean_shape_matching(_voxel: array<Particle, 8>, 
                   _rest_positions: array<vec3<f32>, 8>, 
                   volume: f32, 
                   q_inv: mat3x3<f32>, 
                   material: NeoHookean, 
                   voxel_index: u32,
                   _lagrange_multiplier: LagrangeMultipliers,
                   )
-> ShapeMatchingResult {
    var voxel = _voxel;
    var rest_positions = _rest_positions;
    var lagrange_multiplier = _lagrange_multiplier;
    var com = vec3<f32>(0.0, 0.0, 0.0);
    var total_mass = 0.0;
    var rest_com = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        com += voxel[i].x * voxel[i].mass;
        total_mass += voxel[i].mass;
    }
    com = com / total_mass;
    for (var i = 0; i < 8; i++) {
        // rest_positions[i] += com;
        rest_com += rest_positions[i];
    }
    rest_com = rest_com / 8.0;

    var F = mat3x3<f32>(0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i] - rest_com);
    }
    F = F * q_inv;
    // Compute C_h and C_d
    var C_h = determinant(F) - 1.0;
    var C_d = sqrt(trace(transpose(F) * F)) - sqrt(3.0);
    // Compute gradients
    let ch_fs = mat3x3<f32>(
        cross(F[1], F[2]),
        cross(F[2], F[0]),
        cross(F[0], F[1])
    );

    var grad_C_h = array<vec3<f32>, 8>();
    var grad_C_d = array<vec3<f32>, 8>();
    let r = length(vec3<f32>(length(F[0]), length(F[1]), length(F[2])));
    for (var i = 0; i < 8; i++) {
        grad_C_h[i] = voxel[i].mass * ch_fs * transpose(q_inv) * rest_positions[i];
        grad_C_d[i] = (voxel[i].mass / r) * F * transpose(q_inv) * rest_positions[i];
    }

    // Compute A matrix
    let alpha_h =  1.0 / (volume * material.lambda);
    let alpha_d =  1.0 / (volume * material.mu);
    var A = mat2x2<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        A[0][0] += dot(grad_C_h[i] / voxel[i].mass, grad_C_h[i]);
        A[0][1] += dot(grad_C_h[i] / voxel[i].mass, grad_C_d[i]);
        A[1][0] += dot(grad_C_d[i] / voxel[i].mass, grad_C_h[i]);
        A[1][1] += dot(grad_C_d[i] / voxel[i].mass, grad_C_d[i]);
    }
    let alpha = vec2<f32>(alpha_h, alpha_d) / (uniforms.h * uniforms.h);
    A[0][0] += alpha[0];
    A[1][1] += alpha[1];

    // Compute b vector
    let b = vec2<f32>(-C_h, -C_d);
        //   - vec2<f32>(lagrange_multiplier.lagrange_h * alpha[0], 
        //               lagrange_multiplier.lagrange_d * alpha[1]);

    // Solve 2x2 linear system
    let x = solve_2x2_system(A, b);

    //     let det_A = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    //     let inv_A = mat2x2<f32>(
    //         A[1][1] / det_A, -A[0][1] / det_A,
    //         -A[1][0] / det_A, A[0][0] / det_A
    //     );
    // let x = inv_A * b;
    let delta_lambda = x; // inv_A * b;
    lagrange_multiplier.lagrange_h += delta_lambda[0];
    lagrange_multiplier.lagrange_d += delta_lambda[1];


    for (var i = 0; i < 8; i++) {
        voxel[i].x += (lagrange_multiplier.lagrange_h * grad_C_h[i] + lagrange_multiplier.lagrange_d * grad_C_d[i]) / voxel[i].mass;
    }

    // // Store results for visualization or debugging
    f_matrices[voxel_index].F_d = q_inv;
    f_matrices[voxel_index].lagrange_h = lagrange_multiplier.lagrange_h;
    f_matrices[voxel_index].lagrange_d = lagrange_multiplier.lagrange_d;
    // f_matrices[voxel_index].grad_C_h = grad_C_h;
    // f_matrices[voxel_index].grad_C_d = grad_C_d;
    f_matrices[voxel_index].C_h = C_h;
    f_matrices[voxel_index].C_d = C_d;


    var new_com = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        new_com += voxel[i].x * voxel[i].mass;
    }
    new_com = new_com / total_mass;
    F = mat3x3<f32>(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
    for (var i = 0; i < 8; i++) {
        let dx = voxel[i].x - new_com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i] - rest_com);
    }
    F = F * q_inv;
    for (var i = 0; i < 8; i++) {
        voxel[i].x = (F * rest_positions[i]) + new_com;
    }

    

    f_matrices[voxel_index].F_h = F;

    return ShapeMatchingResult(voxel, lagrange_multiplier);
}

fn decoupled_neohookean_shape_matching(_voxel: array<Particle, 8>, _rest_positions: array<vec3<f32>, 8>, volume: f32, q_inv: mat3x3<f32>, material: NeoHookean, voxel_index: u32, lm: LagrangeMultipliers) -> ShapeMatchingResult {
    var voxel = _voxel;
    var rest_positions = _rest_positions;
    var com = vec3<f32>(0.0);
    var total_mass = 0.0;
    for (var i = 0u; i < 8u; i++) {
        com += voxel[i].x * voxel[i].mass;
        total_mass += voxel[i].mass;
    }
    com /= total_mass;

    var F = mat3x3<f32>(0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0);
    for (var i = 0u; i < 8u; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = uniforms.s * F * q_inv;

    var C_h = determinant(F) - material.gamma;
    var C_d = sqrt(trace(transpose(F) * F));

    var grad_C_h = array<vec3<f32>, 8>();
    var grad_C_d = array<vec3<f32>, 8>();
    let r = sqrt(dot(F[0], F[0]) + dot(F[1], F[1]) + dot(F[2], F[2]));
    let ch_fs = mat3x3<f32>(
        cross(F[1], F[2]),
        cross(F[2], F[0]),
        cross(F[0], F[1])
    ) * transpose(q_inv);
    var sum_grad_C_h_norm = 0.0;
    var sum_grad_C_d_norm = 0.0;
    for (var i = 0u; i < 8u; i++) {
        grad_C_h[i] = uniforms.s * voxel[i].mass * ch_fs * rest_positions[i];
        grad_C_d[i]  = uniforms.s * (voxel[i].mass / r) * F * transpose(q_inv) * rest_positions[i];
        sum_grad_C_h_norm += dot(grad_C_h[i] / voxel[i].mass, grad_C_h[i]);
        sum_grad_C_d_norm += dot(grad_C_d[i] / voxel[i].mass, grad_C_d[i]);
    }

    let alpha_h = 1.0 / (volume * material.lambda);
    let alpha_d = 1.0 / (volume * material.mu);
    
    let lagrange_h = -C_h / (sum_grad_C_h_norm + (alpha_h / (uniforms.h * uniforms.h)));
    let lagrange_d = -C_d / (sum_grad_C_d_norm + (alpha_d / (uniforms.h * uniforms.h)));

    for (var i = 0u; i < 8u; i++) {
        // voxel[i].x += (lagrange_h * grad_C_h[i] + lagrange_d * grad_C_d[i]) / voxel[i].mass;
    }

    com = vec3<f32>(0.0, 0.0, 0.0);
    total_mass = 0.0;
    for (var i = 0u; i < 8u; i++) {
        com += voxel[i].x * voxel[i].mass;
        total_mass += voxel[i].mass;
    }
    com /= total_mass;

    F = mat3x3<f32>(0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0);
    for (var i = 0u; i < 8u; i++) {
        let dx = voxel[i].x - com;
        F += voxel[i].mass * outer_product(dx, rest_positions[i]);
    }
    F = uniforms.s * F * q_inv;

    for (var i = 0u; i < 8u; i++) {
        voxel[i].x = (F * rest_positions[i]) + com;
    }


    return ShapeMatchingResult(voxel, lm);
}



@compute @workgroup_size(256, 1, 1)
fn cube_voxel_shape_matching(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (global_id.x >= uniforms.num_voxels) {
        return;
    }
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
    let rest_length = uniforms.rest_length;
    let h_length = rest_length / 2.0;
    var rest_positions = array<vec3<f32>, 8>(
        vec3(-1.0, -1.0, -1.0) * h_length, // 0b000
        vec3(1.0, -1.0, -1.0) * h_length, // 0b001
        vec3(-1.0, 1.0, -1.0) * h_length, // 0b010
        vec3(1.0, 1.0, -1.0) * h_length, // 0b011
        vec3(-1.0, -1.0, 1.0) * h_length, // 0b100
        vec3(1.0, -1.0, 1.0) * h_length, // 0b101
        vec3(-1.0, 1.0, 1.0) * h_length, // 0b110
        vec3(1.0, 1.0, 1.0) * h_length, // 0b111
    );
    f_matrices[voxel_index].grad_C_h = rest_positions;
    // var rest_com = vec3<f32>(0.0, 0.0, 0.0);
    // for (var i = 0; i < 8; i++) {
    //     rest_com += rest_positions[i];
    // }
    // rest_com /= 8.0;
    // for (var i = 0; i < 8; i++) {
    //     rest_positions[i] -= rest_com;
    // }
    // for a cube, its Q is diag(vec3(volume))
    // so inv(Q) is diag(vec3(1.0 / volume))
    let volume = rest_length * rest_length * rest_length;
    // let q_inv = uniforms.inv_q;
    var Q = mat3x3<f32>(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    );
    for (var i = 0; i < 8; i++) {
        Q += voxel[i].mass * outer_product(rest_positions[i], rest_positions[i]);
    }
    let q_inv = invert(Q);

    let E = 8.0e7; // youngs modulus of rubber
    let nu = 0.4; // poissons ratio of rubber
    let mu = E / (2.0 * (1.0 + nu));
    let lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let material = NeoHookean(lambda, mu, 1.0 + (mu / lambda));

    var lm = LagrangeMultipliers(0.0, 0.0);
    var result = coupled_neohookean_shape_matching(voxel, rest_positions, volume, q_inv, material, voxel_index, lm);
    voxel = result.voxel;
    // var result = shape_matching(voxel, rest_positions, volume, q_inv, material, voxel_index, lm);
    // var result = coupled_neohookean_shape_matching(voxel, rest_positions, volume, q_inv, material, voxel_index, lm);
    // for (var i: u32 = 0; i < 4; i++) {
    //     result = coupled_neohookean_shape_matching(result.voxel, rest_positions, volume, q_inv, material, voxel_index, lm);
    //     lm = result.lagrange_multiplier;
    // }
    // let material = Hookean(E, nu);
    // var result = decoupled_hookean_shape_matching(voxel, rest_positions, volume, q_inv, material, u32(0), lm);

    // var tet_indices = array<array<u32, 4>, 5>(
    //     // array<u32, 4>(0, 1, 2, 4),
    //     // array<u32, 4>(1, 2, 3, 7),
    //     // array<u32, 4>(1, 5, 7, 4),
    //     // array<u32, 4>(6, 2, 7, 4),
    //     array<u32, 4>(1, 2, 4, 7),
    //     array<u32, 4>(0, 3, 5, 6),
    //     array<u32, 4>(0, 1, 2, 3),
    //     array<u32, 4>(4, 7, 5, 6),
    //     array<u32, 4>(2, 3, 1, 7),
    // );

    // for (var i: u32 = 0; i < 5; i++) {
    //     var idxs = tet_indices[i];
    //     var tet_particles = array<Particle, 4>(
    //         voxel[idxs[0]],
    //         voxel[idxs[1]],
    //         voxel[idxs[2]],
    //         voxel[idxs[3]],
    //     );
    //     var tet_rest_positions = array<vec3<f32>, 4>(
    //         rest_positions[idxs[0]],
    //         rest_positions[idxs[1]],
    //         rest_positions[idxs[2]],
    //         rest_positions[idxs[3]],
    //     );

    //     // Compute the volume of the tetrahedron
    //     let edge1 = tet_rest_positions[1] - tet_rest_positions[0];
    //     let edge2 = tet_rest_positions[2] - tet_rest_positions[0];
    //     let edge3 = tet_rest_positions[3] - tet_rest_positions[0];
    //     let tet_volume = abs(dot(edge1, cross(edge2, edge3))) / 6.0;
    //     let total_volume_frac = tet_volume / volume;
    //     var Q = mat3x3<f32>(
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //     );
    //     for (var j = 0; j < 4; j++) {
    //         Q +=  tet_particles[j].mass * outer_product(tet_rest_positions[j], tet_rest_positions[j]);
    //     }
    //     let q_inv = invert(Q);
    //     lm = LagrangeMultipliers(0.0, 0.0);

    //     var result = coupled_tetrahedra_shape_matching(tet_particles, tet_rest_positions, tet_volume, q_inv, material, voxel_index, lm);
    //     voxel[idxs[0]] = result.voxel[0];
    //     voxel[idxs[1]] = result.voxel[1];
    //     voxel[idxs[2]] = result.voxel[2];
    //     voxel[idxs[3]] = result.voxel[3];
    // }

    for (var i: u32 = 0; i < 8; i++) {
        particles[voxel_index * 8 + i] = voxel[i];
    }
    // lagrange_multipliers[voxel_index] = result.lagrange_multiplier;
}

fn apply_face_constraint(_face_constraint: FaceConstraint, constraint_id: u32) {    
    var face_constraint = _face_constraint;
    // Load particles referred by the face constraint
    var face_particles: array<Particle, 8>;
    for (var i = 0u; i < 8u; i++) {
        face_particles[i] = particles[face_constraint.particle_indices[i]];
    }

    // Treat these particles as a voxel and run shape matching
    let rest_length = uniforms.rest_length;
    let h_length = rest_length / 2.0;
    var rest_positions = array<vec3<f32>, 8>(
        vec3(-1.0, -1.0, -1.0) * h_length, // 0b000
        vec3(1.0, -1.0, -1.0) * h_length, // 0b001
        vec3(-1.0, 1.0, -1.0) * h_length, // 0b010
        vec3(1.0, 1.0, -1.0) * h_length, // 0b011
        vec3(-1.0, -1.0, 1.0) * h_length, // 0b100
        vec3(1.0, -1.0, 1.0) * h_length, // 0b101
        vec3(-1.0, 1.0, 1.0) * h_length, // 0b110
        vec3(1.0, 1.0, 1.0) * h_length, // 0b111
    );
    let volume = rest_length * rest_length * rest_length;
    let q_inv = uniforms.inv_q;

    let E = 8.0e8; // youngs modulus of rubber
    let nu = 0.4; // poissons ratio of rubber
    let mu = E / (2.0 * (1.0 + nu));
    let lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let material = NeoHookean(lambda, mu, 1.0 + (mu / lambda));

    var lm = LagrangeMultipliers(0.0, 0.0);
    // var result = shape_matching(face_particles, rest_positions, volume, q_inv, material, u32(0), lm);
    var result = coupled_neohookean_shape_matching(face_particles, rest_positions, volume, q_inv, material, u32(constraint_id + uniforms.num_voxels), lm);
    // for (var i: u32 = 0; i < 4; i++) {
    //     result = coupled_neohookean_shape_matching(result.voxel, rest_positions, volume, q_inv, material, u32(constraint_id + uniforms.num_voxels), lm);
    //     lm = result.lagrange_multiplier;
    // }
    // let material = Hookean(E, nu);
    // var result = decoupled_hookean_shape_matching(voxel, rest_positions, volume, q_inv, material, u32(0), lm);

    // let material = Hookean(E, nu);
    // var result = decoupled_hookean_shape_matching(face_particles, rest_positions, volume, q_inv, material, u32(0), lm);
    // Update the original particles with the shape-matched results
    for (var i = 0u; i < 8u; i++) {
        particles[face_constraint.particle_indices[i]] = result.voxel[i];
    }
    // lagrange_multipliers[uniforms.num_voxels + constraint_id] = result.lagrange_multiplier;
}

@compute @workgroup_size(256, 1, 1)
fn apply_face_constraint_partition(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let constraint_partition = constraint_partitions[current_partition];
    if (global_id.x >= (constraint_partition.end - constraint_partition.start)) {
        return;
    }
    let face_constraint = face_constraints[constraint_partition.start + global_id.x];
    apply_face_constraint(face_constraint, constraint_partition.start + global_id.x);
}

@compute @workgroup_size(1, 1, 1)
fn step_constraint_partition() {
    current_partition = (current_partition + 1) % uniforms.num_constraint_partitions;
}


@compute @workgroup_size(256, 1, 1)
fn boundary_constraints(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= uniforms.num_particles) {
        return;
    }
    let particle_index = global_id.x;
    var particle = particles[particle_index];
    let epsilon = 0.0001; // Small offset to prevent sticking to the boundary

    // Coefficients
    let restitution = 0.1; // No bounce
    let friction = 0.5;    // Higher friction

    // Check each dimension
    for (var dim = 0; dim < 3; dim++) {
        var collision = false;

        // Check for collision and correct position
        if (particle.x[dim] < uniforms.boundary_min[dim] + epsilon) {
            particle.x[dim] = uniforms.boundary_min[dim] + epsilon;
            collision = true;
        } else if (particle.x[dim] > uniforms.boundary_max[dim] - epsilon) {
            particle.x[dim] = uniforms.boundary_max[dim] - epsilon;
            collision = true;
        }

        if (collision) {
            // Calculate the normal vector
            var normal = vec3<f32>(0.0, 0.0, 0.0);
            normal[dim] = 1.0;

            // Compute relative velocity in normal direction
            let v_normal_mag = dot(particle.v, normal);

            // Only adjust if moving into the boundary
            if (v_normal_mag < 0.0) {
                let v_normal = v_normal_mag * normal;
                let v_tangent = particle.v - v_normal;

                // Apply restitution and friction
                let v_normal_new = -v_normal * restitution;
                let v_tangent_new = v_tangent * (1.0 - friction);

                // Update the particle's velocity
                particle.v = v_normal_new + v_tangent_new;
            }
        }
    }

    particles[particle_index] = particle;
}

@compute @workgroup_size(256, 1, 1)
fn update_velocity(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (global_id.x >= uniforms.num_particles) {
        return;
    }
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


@compute @workgroup_size(256, 1, 1)
fn apply_damping(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (global_id.x >= uniforms.num_voxels) {
        return;
    }
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
        let R_i = mat3x3<f32>(
            0.0, -r.z, r.y,
            r.z, 0.0, -r.x,
           -r.y, r.x, 0.0
        );
        inertia_tensor += R_i * transpose(R_i) * voxel[i].mass;
    }
    var angular_velocity = invert(inertia_tensor) * angular_momentum;
    let damping_factor = 0.33; //1.0 / uniforms.h; // Adjust as needed
    for (var i: u32 = 0; i < 8; i++) {
        let r = voxel[i].x - com_position;
        let damped_velocity = com_velocity + cross(angular_velocity, r);
        voxel[i].v = voxel[i].v + min(damping_factor * uniforms.h, 1.0) * (damped_velocity - voxel[i].v);
        particles[voxel_index * 8 + i] = voxel[i];
    }
}