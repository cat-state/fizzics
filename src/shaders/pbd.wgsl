const time_step: f32 = 1.0 / 60.0;
const gravity: vec3<f32> = vec3<f32>(0.0, -9.8, 0.0);
const constraint_stiffness: f32 = 0.5;
const face_constraint_stiffness: f32 = 0.001;
const rest_length: f32 = 0.5;
const collision_damping: f32 = 0.03;
const mouse_attraction_strength: f32 = 10.0;
const num_cubes: i32 = 8*16*2;
const vertices_per_cube: i32 = 8;
const total_vertices: i32 = num_cubes * vertices_per_cube;
const cube_collision_radius: f32 = 0.25;
const boundary: vec3<f32> = vec3<f32>(100.0, 16.0, 100.0);
const boundary_offset: vec3<f32> = vec3<f32>(0.0, -4.0, 0.0);
const solve_iterations: u32 = 1;
struct Particle {
    x: vec3<f32>,
    mass: f32,
    v: vec3<f32>,
    _padding: f32,
    q: vec4<f32>
}

struct Voxel {
    particles: array<Particle, 8>,
}

// left indices, right indices
// each pair of edges between two neighbouring verts constraints the opposing vertex 
// on the other face
struct FaceConstraint {
    l: array<i32, 4>,
    r: array<i32, 4>,
}

struct Uniforms {
    i_mouse: vec4<f32>,
    i_resolution: vec2<f32>,
    i_frame: i32,
    constraint_phase: u32,
    i_offset: u32
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
@group(0) @binding(2) var<storage, read_write> face_constraints: array<FaceConstraint>;

fn extract_rotation(A: mat3x3<f32>, q: vec4<f32>, max_iter: u32) -> vec4<f32> {
    var q_out = q;
    for (var iter: u32 = 0u; iter < max_iter; iter++) {
        let R = quaternion_to_matrix(q_out);
        
        let omega = (
            cross(R[0], A[0]) + 
            cross(R[1], A[1]) + 
            cross(R[2], A[2])
        ) * (1.0 / (abs(dot(R[0], A[0]) + dot(R[1], A[1]) + dot(R[2], A[2])) + 1.0e-9));
        
        let w = length(omega);
        if (w < 1.0e-9) {
            break;
        }
        
        let axis = normalize(omega);
        let angle_axis = vec4<f32>(axis * sin(w * 0.5), cos(w * 0.5));
        q_out = quaternion_multiply(angle_axis, q_out);
        q_out = normalize(q_out);
    }
    return q_out;
}

fn quaternion_to_matrix(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        vec3<f32>(2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        vec3<f32>(2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))
    );
}

fn quaternion_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}



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

fn apply_gram_schmidt_face_constraint(_C: FaceConstraint) {

    var C = _C;
    var l_signs = array<f32, 4>(sign(f32(C.l[0])), sign(f32(C.l[1])), sign(f32(C.l[2])), sign(f32(C.l[3])));
    var r_signs = array<f32, 4>(sign(f32(C.r[0])), sign(f32(C.r[1])), sign(f32(C.r[2])), sign(f32(C.r[3])));

    for (var i = 0; i < 4; i++) {
        C.l[i] = abs(C.l[i]);
        C.r[i] = abs(C.r[i]);
    }

    var l_particles = array<Particle, 4>(
        particles[C.l[0]],
        particles[C.l[1]],
        particles[C.l[2]],
        particles[C.l[3]],
    );

    var r_particles = array<Particle, 4>(
        particles[C.r[0]],
        particles[C.r[1]],
        particles[C.r[2]],
        particles[C.r[3]],
    );

    var ideal_ls = array<vec3<f32>, 4>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var ideal_rs = array<vec3<f32>, 4>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    for (var step = 0; step < 1; step++) {
        for (var i = 0; i < 4; i++) {
            let l_next_edge = l_particles[(i + 1) % 4].x - l_particles[i].x;
            let l_prev_edge = l_particles[(i + 3) % 4].x - l_particles[i].x;
            let r_next_edge = r_particles[(i + 1) % 4].x - r_particles[i].x;
            let r_prev_edge = r_particles[(i + 3) % 4].x - r_particles[i].x;

            let r_pred = (l_signs[i] * cross(l_next_edge, l_prev_edge));
            let l_pred = (r_signs[i] * cross(r_next_edge, r_prev_edge));

            ideal_ls[i] = l_pred;
            ideal_rs[i] = r_pred;

            let l_correction = (ideal_ls[i] - l_particles[i].x) * face_constraint_stiffness;
            let r_correction = (ideal_rs[i] - r_particles[i].x) * face_constraint_stiffness;

            let l_old_x = l_particles[i].x;
            let r_old_x = r_particles[i].x;
            l_particles[i].x += l_correction;
            r_particles[i].x += r_correction;

        }

        for (var i = 0; i < 4; i++) {
            let l_correction = (ideal_ls[i] - l_particles[i].x) * face_constraint_stiffness;
            let r_correction = (ideal_rs[i] - r_particles[i].x) * face_constraint_stiffness;

            let l_old_x = l_particles[i].x;
            let r_old_x = r_particles[i].x;
            // l_particles[i].x += l_correction;
            // r_particles[i].x += r_correction;
            // l_particles[i].v = (l_particles[i].x - l_old_x) / time_step;
            // r_particles[i].v = (r_particles[i].x - r_old_x) / time_step;
        }
    }

    particles[C.l[0]] = l_particles[0];
    particles[C.l[1]] = l_particles[1];
    particles[C.l[2]] = l_particles[2];
    particles[C.l[3]] = l_particles[3];
    particles[C.r[0]] = r_particles[0];
    particles[C.r[1]] = r_particles[1];
    particles[C.r[2]] = r_particles[2];
    particles[C.r[3]] = r_particles[3];
}



fn quaternion_from_matrix(m: mat3x3<f32>) -> vec4<f32> {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let s = sqrt(max(1.0 + trace, 0.0));
    let w = 0.5 * s;
    let x = 0.5 * (m[2][1] - m[1][2]) / s;
    let y = 0.5 * (m[0][2] - m[2][0]) / s;
    let z = 0.5 * (m[1][0] - m[0][1]) / s;
    return vec4<f32>(x, y, z, w);
}

fn apply_gram_schmidt_constraint(_voxel: Voxel) -> Voxel {
    var voxel = _voxel;
    var corrections = array<vec3<f32>, 8>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var n_corrections = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var _step: u32 = 0; _step < solve_iterations; _step++) {
        for (var _i: u32 = 0; _i < u32(vertices_per_cube); _i++) {
            var i = (_i + uniforms.i_offset) % u32(vertices_per_cube);
            let next1 = (i + 1) % 4 + (i / 4) * 4;
            let next2 = (i + 3) % 4 + (i / 4) * 4;
            let next3 = i ^ 4;

            let edge1 = voxel.particles[next1].x - voxel.particles[i].x;
            let edge2 = voxel.particles[next2].x - voxel.particles[i].x;
            let edge3 = voxel.particles[next3].x - voxel.particles[i].x;

            // let edgem = transpose(mat3x3<f32>(edge1, edge2, edge3));
            // var q: vec4<f32> = voxel.particles[i].q;
            // if(q[3] == 0.0) {
            //     // Convert the edge matrix to a quaternion
            //     let quat_from_matrix = quaternion_from_matrix(edgem);
            
            //     // Normalize the quaternion
            //     q = normalize(quat_from_matrix);
            // }

            // q = extract_rotation(edgem, q, u32(10));
            // voxel.particles[i].q = q;
            
            // let us = quaternion_to_matrix(q);
            // let u1 = normalize(us[0]) * rest_length;
            // let u2 = normalize(us[1]) * rest_length;
            // let u3 = normalize(us[2]) * rest_length;
            // let u1 = normalize(vec3<f32>(us[0][0], us[1][0], us[2][0])) * rest_length;
            // let u2 = normalize(vec3<f32>(us[0][1], us[1][1], us[2][1])) * rest_length;
            // let u3 = normalize(vec3<f32>(us[0][2], us[1][2], us[2][2])) * rest_length;
            var u1: vec3<f32> = edge1;
            var u2: vec3<f32> = edge2;
            var u3: vec3<f32> = edge3;

            for(var substep = 0; substep < 1; substep++) {
                let gs_0 = gram_schmidt(u1, u2, u3);
                let gs_1 = gram_schmidt(u2, u3, u1);
                let gs_2 = gram_schmidt(u3, u1, u2);

                // u1 = average_on_sphere(u1, gs[1], gs[2]);
            // u2 = average_on_sphere(u2, gs[2], gs[0]);
            // u3 = average_on_sphere(u3, gs[0], gs[1]);

                u1 = normalize(gs_0[0] + gs_2[1] + gs_1[2]) * rest_length;
                u2 = normalize(gs_0[1] + gs_1[0] + gs_2[2]) * rest_length;
                u3 = normalize(gs_0[2] + gs_1[1] + gs_2[0]) * rest_length;
            }

            // let u4 = normalize(gs_3[0] + gs_5[1] + gs_4[2]) * rest_length;
            // let u5 = normalize(gs_3[2] + gs_4[0] + gs_5[1]) * rest_length;
            // let u6 = normalize(gs_3[1] + gs_4[2] + gs_5[0]) * rest_length;
            // let u1 = average_on_sphere(gs_0[0], gs_2[1], gs_1[2]) * rest_length;
            // let u2 = average_on_sphere(gs_0[1], gs_1[0], gs_2[2]) * rest_length;
            // let u3 = average_on_sphere(gs_0[2], gs_1[1], gs_2[0]) * rest_length;

            let ideal_next1 = voxel.particles[i].x + u1;
            let ideal_next2 = voxel.particles[i].x + u2;
            let ideal_next3 = voxel.particles[i].x + u3;

            let correction_next1 = (ideal_next1 - voxel.particles[next1].x);
            let correction_next2 = (ideal_next2 - voxel.particles[next2].x);
            let correction_next3 = (ideal_next3 - voxel.particles[next3].x);

            // this has no translational bias but causes voxels to spin 
            let total_correction = correction_next1 + correction_next2 + correction_next3;
            // let correction_self = -total_correction / 4.0;
            // voxel.particles[next1].x += correction_next1 * 0.25;
            // voxel.particles[next2].x += correction_next2 * 0.25;
            // voxel.particles[next3].x += correction_next3 * 0.25;
            // voxel.particles[i].x += correction_self;

            let correction_self = -total_correction / 3.0;

            corrections[next1] += correction_next1;
            corrections[next2] += correction_next2;
            corrections[next3] += correction_next3;
            corrections[i] += correction_self;

            n_corrections[next1] += 1.0;
            n_corrections[next2] += 1.0;
            n_corrections[next3] += 1.0;
            n_corrections[i] += 1.0;
            // let correction_self = -(correction_next1 + correction_next2 + correction_next3) / 3.0;

            // voxel.particles[next1].x += correction_next1;
            // voxel.particles[next2].x += correction_next2;
            // voxel.particles[next3].x += correction_next3;
            // voxel.particles[i].x += correction_self;
        }

        for (var i = 0; i < vertices_per_cube; i++) {
            if(n_corrections[i] > 0.0) {
                voxel.particles[i].x += constraint_stiffness * corrections[i] / n_corrections[i];
                corrections[i] = vec3<f32>(0.0);
                n_corrections[i] = 0.0;
            }
        }
    }
    return voxel;
}

fn handle_boundary_collisions(_particle: Particle, prev_pos: vec3<f32>) -> Particle {
    var particle = _particle;
    var velocity = (particle.x - prev_pos);
    if ((particle.x.x + boundary_offset.x) < -boundary.x) {
        particle.x.x = -boundary.x - boundary_offset.x;
        velocity.x *= -collision_damping;
    } else if ((particle.x.x + boundary_offset.x) > boundary.x) {
        particle.x.x = boundary.x - boundary_offset.x;
        velocity.x *= -collision_damping;
    }

    if ((particle.x.y + boundary_offset.y) < -boundary.y) {
        particle.x.y = -boundary.y - boundary_offset.y;
        velocity.y *= -collision_damping;
    } else if ((particle.x.y + boundary_offset.y) > boundary.y) {
        particle.x.y = boundary.y - boundary_offset.y;
        velocity.y *= -collision_damping;
    }

    if ((particle.x.z + boundary_offset.z) < -boundary.z) {
        particle.x.z = -boundary.z - boundary_offset.z;
        velocity.z *= -collision_damping;
    } else if ((particle.x.z + boundary_offset.z) > boundary.z) {
        particle.x.z = boundary.z - boundary_offset.z;
        velocity.z *= -collision_damping;
    }
    particle.v = velocity / time_step;
    return particle;
}

@compute @workgroup_size(1, 1, 1)
fn handle_particle_collisions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    var p_i = particles[particle_index];
    let ori_pos = p_i.x;
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

@compute @workgroup_size(1, 1, 1)
fn voxel_constraint(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cube_index = global_id.x;
    
    var voxel = get_voxel(cube_index);
    let mouse_pos = vec3<f32>((uniforms.i_mouse.xy - 0.5 * uniforms.i_resolution) / uniforms.i_resolution.y, 0.0);

    var closest_index = -1;
    var h = time_step;
    var prev_poses = array<vec3<f32>, 8>(voxel.particles[0].x, voxel.particles[1].x, voxel.particles[2].x, voxel.particles[3].x, voxel.particles[4].x, voxel.particles[5].x, voxel.particles[6].x, voxel.particles[7].x);

    for (var i = 0; i < vertices_per_cube; i++) {
        var particle = voxel.particles[i];
        var curr_pos = particle.x;
        particle.v *= 0.99;
        var velocity = particle.v;

        var new_pos = curr_pos + velocity * h + gravity * h * h;
        particle.x = new_pos;
        particle = handle_boundary_collisions(particle, curr_pos);
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
        var curr_pos = particle.x;
        particle = handle_boundary_collisions(particle, curr_pos);
        particles[global_particle_index] = particle;
    }
}

@compute @workgroup_size(1, 1, 1)
fn apply_x_face_constraints(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let constraint_index = global_id.x;
    let n_constraints = num_workgroups.x;
    let C = face_constraints[constraint_index + n_constraints * 0];
    // apply_gram_schmidt_face_constraint(C);

    var voxel = Voxel(
        array<Particle, 8>(
            particles[abs(C.l[0])],
            particles[abs(C.l[1])],
            particles[abs(C.l[2])],
            particles[abs(C.l[3])],
            particles[abs(C.r[0])],
            particles[abs(C.r[1])],
            particles[abs(C.r[2])],
            particles[abs(C.r[3])],
        )
    );

    voxel = apply_gram_schmidt_constraint(voxel);

    particles[abs(C.l[0])] = voxel.particles[0];
    particles[abs(C.l[1])] = voxel.particles[1];
    particles[abs(C.l[2])] = voxel.particles[2];
    particles[abs(C.l[3])] = voxel.particles[3];
    particles[abs(C.r[0])] = voxel.particles[4];
    particles[abs(C.r[1])] = voxel.particles[5];
    particles[abs(C.r[2])] = voxel.particles[6];
    particles[abs(C.r[3])] = voxel.particles[7];
}

@compute @workgroup_size(1, 1, 1)
fn apply_y_face_constraints(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let constraint_index = global_id.x;
    let n_constraints = num_workgroups.x;
    let C = face_constraints[constraint_index + n_constraints * 1];
    // apply_gram_schmidt_face_constraint(C);


    var voxel = Voxel(
        array<Particle, 8>(
            particles[abs(C.l[0])],
            particles[abs(C.l[1])],
            particles[abs(C.l[2])],
            particles[abs(C.l[3])],
            particles[abs(C.r[0])],
            particles[abs(C.r[1])],
            particles[abs(C.r[2])],
            particles[abs(C.r[3])],
        )
    );

    voxel = apply_gram_schmidt_constraint(voxel);

    particles[abs(C.l[0])] = voxel.particles[0];
    particles[abs(C.l[1])] = voxel.particles[1];
    particles[abs(C.l[2])] = voxel.particles[2];
    particles[abs(C.l[3])] = voxel.particles[3];
    particles[abs(C.r[0])] = voxel.particles[4];
    particles[abs(C.r[1])] = voxel.particles[5];
    particles[abs(C.r[2])] = voxel.particles[6];
    particles[abs(C.r[3])] = voxel.particles[7];

}


@compute @workgroup_size(1, 1, 1)
fn apply_z_face_constraints(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let constraint_index = global_id.x;
    let n_constraints = num_workgroups.x;
    let C = face_constraints[constraint_index + n_constraints * 2];
    // apply_gram_schmidt_face_constraint(C);



    var voxel = Voxel(
        array<Particle, 8>(
            particles[abs(C.l[0])],
            particles[abs(C.l[1])],
            particles[abs(C.l[2])],
            particles[abs(C.l[3])],
            particles[abs(C.r[0])],
            particles[abs(C.r[1])],
            particles[abs(C.r[2])],
            particles[abs(C.r[3])],
        )
    );

    voxel = apply_gram_schmidt_constraint(voxel);

    particles[abs(C.l[0])] = voxel.particles[0];
    particles[abs(C.l[1])] = voxel.particles[1];
    particles[abs(C.l[2])] = voxel.particles[2];
    particles[abs(C.l[3])] = voxel.particles[3];
    particles[abs(C.r[0])] = voxel.particles[4];
    particles[abs(C.r[1])] = voxel.particles[5];
    particles[abs(C.r[2])] = voxel.particles[6];
    particles[abs(C.r[3])] = voxel.particles[7];

}
