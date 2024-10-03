use std::{borrow::Cow, default};
use std::sync::Arc;
use tao::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};
use wgpu::util::DeviceExt;
use nalgebra as na;
use na::allocator::{Allocator};
use na::base::{DefaultAllocator};
use rand::Rng;
use futures_intrusive;
use ordered_float::NotNan;
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    x: na::Vector3<f32>,
    mass: f32,
    v: na::Vector3<f32>,
    _padding: f32,
    x_prev: na::Vector3<f32>,
    _padding2: f32,
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Voxel {
    particles: [Particle; 8],
}

// 8 particles per pair of voxel faces
// left indices, right indices
// each pair of edges between two neighbouring verts constraints the opposing vertex 
// on the other face
// if idx is negative then the predicted pos is negated

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FaceConstraint {
   idxs: [u32; 8]
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct ConstraintPartition {
    start: u32,
    end: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Force {
    f: na::Vector3<f32>,
    particle_index: i32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    h: f32,
    num_particles: u32,
    num_voxels: u32,
    num_constraint_partitions: u32,
    boundary_min: na::Vector3<f32>,
    s: f32,
    boundary_max: na::Vector3<f32>,
    particle_radius: f32,
    rest_length: f32, 
    _padding2: [f32; 3],
    inv_q: [[f32; 4]; 3],
}

// Add this struct to match the shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FMatrices {
    F_h: [[f32; 4]; 3],
    F_d: [[f32; 4]; 3],
    lagrange_h: f32,
    lagrange_d: f32,
    C_h: f32,
    C_d: f32,
    grad_C_h: [[f32; 4]; 8],
    grad_C_d: [[f32; 4]; 8],
}

// Add this near the top of the file, after other struct definitions
struct ComputeConfig {
    workgroup_size: u32,
}

fn invert(m: na::Matrix3<f32>) -> na::Matrix3<f32> {
    let a = m[(0, 0)];
    let b = m[(0, 1)];
    let c = m[(0, 2)];
    let d = m[(1, 0)];
    let e = m[(1, 1)];
    let f = m[(1, 2)];
    let g = m[(2, 0)];
    let h = m[(2, 1)];
    let i = m[(2, 2)];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;

    na::Matrix3::new(
        (e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det
    )
}


fn cube_inv_Q(voxel: Voxel) -> (na::Matrix3<f32>, f32) {
    let total_mass = voxel.particles.iter().map(|p| p.mass as f32).sum::<f32>();
    let com = voxel.particles.iter().map(|p| p.x.cast::<f32>()).reduce(|a, b| a + b).unwrap() / 8.0f32;
    let rest_positions = voxel.particles.iter().map(|p| p.x.cast::<f32>() - com).collect::<Vec<_>>();
    dbg!(&rest_positions);
    let Q = voxel.particles.iter().map(|p| {
        let r_i_bar = p.x.cast::<f32>() - com;
        (p.mass as f32) * r_i_bar * r_i_bar.transpose()
    }).reduce(|a, b| a + b).unwrap();
    dbg!(&Q);
    let s = 1.0; //Q.sum();
    (invert(s * Q), s)
}

fn voxel_cube(size: na::Vector3<i32>, rest_length: f32) -> (Vec<Voxel>, (Vec<FaceConstraint>, Vec<FaceConstraint>, Vec<FaceConstraint>)) {
    /*
     *     6 ---- y
     *    ..     .|
     *   . .    . |
     *  2 ---- 3  |
     *  . \.   |  |
     *  .  4 --|- 5
     *  . .    |.
     *  0 ---- 1
     */
    // let x_faces = [[1, 2, 3, 2], [5, 4, 7, 6]];
    // let y_faces = [[3, 2, 1, 0], [7, 6, 5, 4]];
    // let z_faces = [[4, 5, 6, 7], [0, 1, 2, 3]];
    let flip_x = 0b001;
    let flip_y = 0b010;
    let flip_z = 0b100;
    let offsets = [
        na::Vector3::new(-1.0, -1.0, -1.0), // 0b000
        na::Vector3::new(1.0, -1.0, -1.0),  // 0b001
        na::Vector3::new(-1.0, 1.0, -1.0),  // 0b010
        na::Vector3::new(1.0, 1.0, -1.0),   // 0b011
        na::Vector3::new(-1.0, -1.0, 1.0),  // 0b100
        na::Vector3::new(1.0, -1.0, 1.0),   // 0b101
        na::Vector3::new(-1.0, 1.0, 1.0),   // 0b110
        na::Vector3::new(1.0, 1.0, 1.0),     // 0b111
    ];
    let zero = na::Vector3::zeros();
    let center = na::Vector3::new( size.x as f32, 0.0f32, size.z as f32);

    let mut voxels = Vec::new();
    let mut x_constraints = Vec::new();
    let mut y_constraints = Vec::new();
    let mut z_constraints = Vec::new();
    for x in 0..size.x {
        for y in 0..size.y {
            for z in 0..size.z {
                let p = na::Vector3::new(x as f32, y as f32, z as f32);
                let voxel = Voxel {
                    particles: offsets.map(|offset| 
                        Particle { 
                            x: 2.0 * rest_length * p + offset * (rest_length / 2.0) - center, 
                            mass: 1.0,
                            v: zero,
                            _padding: 0.0,
                            x_prev: na::Vector3::zeros(),
                            _padding2: 0.0
                        }
                    )
                };
                voxels.push(voxel);
                let midx = x * size.y * size.z + y * size.z + z;
                let mpidx = (midx * 8) as u32;
                if x != 0 {    
                    let vidx = (x - 1) * size.y * size.z + y * size.z + z;
                    let vpidx = (vidx * 8) as u32;
                    let c = FaceConstraint {
                        idxs: [vpidx ^ flip_x, mpidx, vpidx ^ flip_x ^ flip_y, mpidx ^ flip_y,
                               vpidx ^ flip_x ^ flip_z, mpidx ^ flip_z, vpidx ^ flip_x ^ flip_y ^ flip_z, mpidx ^ flip_y ^ flip_z]
                    };
                    x_constraints.push(c);
                }
                if y != 0 {
                    let vidx = x * size.y * size.z + (y - 1) * size.z + z;
                    let vpidx = (vidx * 8) as u32;
                    let c = FaceConstraint {
                        idxs: [vpidx ^ flip_y, vpidx ^ flip_x ^ flip_y, mpidx, mpidx ^ flip_x,
                               vpidx ^ flip_y ^ flip_z, vpidx ^ flip_x ^ flip_y ^ flip_z, mpidx ^ flip_z, mpidx ^ flip_x ^ flip_z]
                    };
                    y_constraints.push(c);
                }
                if z != 0 {
                    let vidx = x * size.y * size.z + y * size.z + (z - 1);
                    let vpidx = (vidx * 8) as u32;
                    let c = FaceConstraint {
                        idxs: [vpidx ^ flip_z, vpidx ^ flip_x ^ flip_z, vpidx ^ flip_y ^ flip_z, vpidx ^ flip_x ^ flip_y ^ flip_z,
                               mpidx, mpidx ^ flip_x, mpidx ^ flip_y, mpidx ^ flip_x ^ flip_y]
                    };
                    z_constraints.push(c);
                }
            }
        }
    }
    (voxels, (x_constraints, y_constraints, z_constraints))
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertex {
    position: na::Vector3<f32>,
    normal: na::Vector3<f32>
}

impl MeshVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}

fn icosahedron_mesh(radius: f32) -> Vec<MeshVertex> {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let scale = 1.0 / (1.0 + t * t).sqrt();

    let vertices = [
        na::Vector3::new(t, 1.0, 0.0) * scale,
        na::Vector3::new(-t, 1.0, 0.0) * scale,
        na::Vector3::new(t, -1.0, 0.0) * scale,
        na::Vector3::new(-t, -1.0, 0.0) * scale,
        na::Vector3::new(1.0, 0.0, t) * scale,
        na::Vector3::new(1.0, 0.0, -t) * scale,
        na::Vector3::new(-1.0, 0.0, t) * scale,
        na::Vector3::new(-1.0, 0.0, -t) * scale,
        na::Vector3::new(0.0, t, 1.0) * scale,
        na::Vector3::new(0.0, -t, 1.0) * scale,
        na::Vector3::new(0.0, t, -1.0) * scale,
        na::Vector3::new(0.0, -t, -1.0) * scale,
    ];

    let triangles = [
        [0, 8, 4], [0, 5, 10], [2, 4, 9], [2, 11, 5], [1, 6, 8],
        [1, 10, 7], [3, 9, 6], [3, 7, 11], [0, 10, 8], [1, 8, 10],
        [2, 9, 11], [3, 11, 9], [4, 2, 0], [5, 0, 2], [6, 1, 3],
        [7, 3, 1], [8, 6, 4], [9, 4, 6], [10, 5, 7], [11, 7, 5],
    ];

    triangles.iter().flat_map(|&[a, b, c]| {
        let va = vertices[a] * radius;
        let vb = vertices[b] * radius;
        let vc = vertices[c] * radius;
        let normal = (vb - va).cross(&(vc - va)).normalize();

        [
            MeshVertex { position: va, normal },
            MeshVertex { position: vb, normal },
            MeshVertex { position: vc, normal },
        ]
    }).collect()
}

fn subdivide_mesh(mesh: Vec<MeshVertex>) -> Vec<MeshVertex> {
    mesh.chunks(3).flat_map(|triangle| {
        let a = triangle[0].position;
        let b = triangle[1].position;
        let c = triangle[2].position;

        let ab = (a + b) / 2.0;
        let bc = (b + c) / 2.0;
        let ca = (c + a) / 2.0;

        let normal = (ab - a).cross(&(bc - a)).normalize();

        [
            MeshVertex { position: a, normal },
            MeshVertex { position: ab, normal },
            MeshVertex { position: ca, normal },

            MeshVertex { position: ab, normal },
            MeshVertex { position: b, normal },
            MeshVertex { position: bc, normal },

            MeshVertex { position: bc, normal },
            MeshVertex { position: c, normal },
            MeshVertex { position: ca, normal },

            MeshVertex { position: ab, normal },
            MeshVertex { position: bc, normal },
            MeshVertex { position: ca, normal },
        ].into_iter()
    }).collect()
}


fn icosahedron_sphere(radius: f32, subdivisions: u32) -> Vec<MeshVertex> {
    let mesh = (0..subdivisions).fold(icosahedron_mesh(radius), |mesh, _| subdivide_mesh(mesh));
    mesh.iter().map(|v| {
        let normal = v.position.normalize();
        MeshVertex { position: normal * radius, normal }
    }).collect()
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawParticlesUniforms {
    view_projection: na::Matrix4<f32>,
    camera_position: na::Vector4<f32>,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let window = Arc::new(window);
    let size = window.inner_size();

    let bw = &window;
    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(window.clone()) }.unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty().union(wgpu::Features::POLYGON_MODE_LINE).union(wgpu::Features::SUBGROUP),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let xpbd_pbsm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("XPBD PBSM"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/xpbd-pbsm.wgsl").into()),
    });

    let draw_particles_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Draw Particles Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/draw_particles.wgsl").into()),
    });


    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    // In the main function, add this before creating the compute pipelines
    let compute_config = ComputeConfig {
        workgroup_size: 256, // Now tunable from the Rust side
    };

    // Update the compute pipeline creations (remove the workgroup size from entry points)
    let voxel_constraints_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Voxel Constraints Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &xpbd_pbsm,
        entry_point: "cube_voxel_shape_matching",
        compilation_options: Default::default(),
        cache: None,
    });

    let apply_face_constraint_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Apply Face Constraint Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &xpbd_pbsm,
        entry_point: "apply_face_constraint_partition",
        compilation_options: Default::default(),
        cache: None,
    });

    let step_constraint_partition_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Step Constraint Partition Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &xpbd_pbsm,
        entry_point: "step_constraint_partition",
        compilation_options: Default::default(),
        cache: None,
    });

    let apply_velocity_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Apply Velocity Forces Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &xpbd_pbsm,
        entry_point: "apply_velocity_forces",
        compilation_options: Default::default(),
        cache: None,
    });

    let update_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Update Velocity Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &xpbd_pbsm,
        entry_point: "update_velocity",
        compilation_options: Default::default(),
        cache: None,
    });

    let boundary_constraints_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Boundary Constraints Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &xpbd_pbsm,
        entry_point: "boundary_constraints",
        compilation_options: Default::default(),
        cache: None,
    });

    let apply_damping_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Apply Damping Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &xpbd_pbsm,
        entry_point: "apply_damping",
        compilation_options: Default::default(),
        cache: None,
    });

    let cube_size = na::Vector3::<i32>::new(2, 2, 2);

    let rest_length = 1.011f32;
    let wood_density = 1520.0;
    let voxel_mass = rest_length * rest_length * rest_length * wood_density;
    let particle_mass = voxel_mass / 8.0;
    let (voxels, xyz_constraints) = voxel_cube(cube_size, rest_length);


    let voxels = voxels.into_iter().map(|mut voxel| {
        let num_particles = voxel.particles.len();
        for particle in voxel.particles.iter_mut() {
            particle.mass = particle_mass; //mass / num_particles as f32;
        }
        voxel
    }).collect::<Vec<Voxel>>();
    let (iQ, s) = cube_inv_Q(voxels[5]);
    dbg!(&iQ);
    dbg!(&s);

    let mut rng = rand::thread_rng();
    //Rotate voxels by 45 degrees around y-axis and x-axis
    let rotation_y = na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), std::f32::consts::FRAC_PI_4);
    let rotation_z = na::Rotation3::from_axis_angle(&na::Vector3::z_axis(), std::f32::consts::FRAC_PI_4);
    let rotation_x = na::Rotation3::from_axis_angle(&na::Vector3::x_axis(), std::f32::consts::FRAC_PI_4);
    let rotation = rotation_x * rotation_y;

    // Calculate the diagonal length of the cube
    let cube_size = 2.0 * rest_length * na::Vector3::<f32>::new(cube_size.x as f32, cube_size.y as f32, cube_size.z as f32);
    let diagonal_length = cube_size.magnitude();
    let y_offset = diagonal_length / 2.0;

    let voxels = voxels.into_iter().map(|mut voxel| {
        for particle in voxel.particles.iter_mut() {
            particle.x = rotation_x * particle.x;
        }
        voxel
    }).collect::<Vec<Voxel>>();
    let lowest_particle = voxels.iter().flat_map(|voxel| voxel.particles.iter()).min_by_key(|p| NotNan::new(p.x.y).unwrap()).unwrap().clone();
    let voxels = voxels.into_iter().map(|mut voxel| {

        for particle in voxel.particles.iter_mut() {
            particle.x.y -= lowest_particle.x.y;
            // let rs = 1.0f32 * na::Vector3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
            // particle.x += rs;
        }
        voxel
    }).collect::<Vec<Voxel>>();

    let constraints = vec![xyz_constraints.0, xyz_constraints.1, xyz_constraints.2];
    let constraint_partitions = constraints.iter().scan(0, |acc, x| {
        let start = *acc;
        *acc += x.len() as u32;
        Some(ConstraintPartition { start, end: *acc })
    }).collect::<Vec<ConstraintPartition>>();
    let constraint_partitions = constraint_partitions.into_iter().filter(|c| c.start != c.end).collect::<Vec<ConstraintPartition>>();
    let num_constraint_partitions = constraint_partitions.len() as u32;
    let num_constraints = constraint_partitions.iter().map(|c| c.end - c.start).sum::<u32>() as usize;
    let num_voxels = voxels.len();
    let num_particles = num_voxels * 8;
    dbg!( bytemuck::cast_slice::<Voxel, u8>(&voxels).len());
    let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particles Buffer"),
        contents: bytemuck::cast_slice(&voxels),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let flat_constraints = constraints.concat();
    let constraint_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Constraints Buffer"),
        contents: bytemuck::cast_slice(&flat_constraints),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let constraint_partitions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Constraint Partitions Buffer"),
        contents: bytemuck::cast_slice(&constraint_partitions),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let lagrange_multipliers = vec![0.0f32; (num_voxels + num_constraints) * 2];
    let lagrange_multipliers_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Lagrange Multipliers Buffer"),
        contents: bytemuck::cast_slice(&lagrange_multipliers),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let forces = vec![Force {
        f: na::Vector3::<f32>::new(0.0, -9.8 * particle_mass, 0.0),
        particle_index: -1,
    }];

    let force_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Forces Buffer"),
        contents: bytemuck::cast_slice(&forces),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let iQ = iQ.cast::<f32>();

    let mut uniforms = Uniforms {
        h: (1.0f32 / 100.0f32) / 20.0f32,
        num_particles: num_particles as u32,
        num_voxels: num_voxels as u32,
        num_constraint_partitions: num_constraint_partitions,
        boundary_min: na::Vector3::<f32>::new(-3200.0, -4.0, -3200.0),
        s: s as f32,
        boundary_max: na::Vector3::<f32>::new(3200.0, 4000.0, 3200.0),
        particle_radius: 0.1,
        rest_length,
        _padding2: [0.0; 3],
        inv_q: [
            [iQ[0], iQ[1], iQ[2], 0.0],
            [iQ[3], iQ[4], iQ[5], 0.0],
            [iQ[6], iQ[7], iQ[8], 0.0],
        ]
    };

    let pbd_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PBD Uniforms Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // In the main function, create a new buffer for F matrices
    let f_matrices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("F Matrices Buffer"),
        size: (std::mem::size_of::<FMatrices>() * (num_voxels + num_constraints)) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let current_partition = 0u32;

    let current_partition_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Current Partition Buffer"),
        contents: bytemuck::cast_slice(&[current_partition]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &compute_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: force_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pbd_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: f_matrices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: constraint_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: constraint_partitions_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: current_partition_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: lagrange_multipliers_buffer.as_entire_binding(),
            },
        ],
    });

    let aspect_ratio = size.width as f32 / size.height as f32;
    let projection = na::Perspective3::new(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 1000.0);
    let view = na::Isometry3::look_at_rh(
        &na::Point3::new(10.0, 10.0, 10.0),
        &na::Point3::new(0.0, 0.0, 0.0),
        &na::Vector3::y(),
    );
    let view_projection = projection.as_matrix() * view.to_homogeneous();
    let camera_position = na::Vector4::new(5.0, 5.0, 5.0, 1.0);

    let draw_particles_uniforms = DrawParticlesUniforms {
        view_projection,
        camera_position,
    };
    let draw_particles_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Draw Particles Uniforms Buffer"),
        contents: bytemuck::cast_slice(&[draw_particles_uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let ico_mesh = icosahedron_sphere(rest_length / 2.1, 1);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sphere Mesh Vertex Buffer"),
        contents: bytemuck::cast_slice(&ico_mesh),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Render Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create a buffer for face constraints

    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Render Bind Group"),
        layout: &render_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: draw_particles_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: constraint_buffer.as_entire_binding(),
            },
        ],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&render_bind_group_layout],
        push_constant_ranges: &[],
    });

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let mut depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let mut depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &draw_particles_shader,
            entry_point: "vertex_main",
            buffers: &[MeshVertex::desc()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &draw_particles_shader,
            entry_point: "fragment_main",
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Wireframe Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &draw_particles_shader,
            entry_point: "vertex_main_wireframe",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &draw_particles_shader,
            entry_point: "fragment_main_wireframe",
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Line,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let mut i_offset: u32 = 0;

    let mut t0 = std::time::Instant::now();

    // Add this after creating the particle_buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (std::mem::size_of::<Particle>() * num_particles) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Add this after creating the staging_buffer
    let f_matrices_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("F Matrices Staging Buffer"),
        size: (std::mem::size_of::<FMatrices>() * (num_voxels + num_constraints)) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                config.width = new_size.width;
                config.height = new_size.height;

                depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Depth Texture"),
                    size: wgpu::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
            
                depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    
                surface.configure(&device, &config);
                window.request_redraw();
            },
            Event::MainEventsCleared => {
                use std::thread;
                // thread::sleep(std::time::Duration::from_millis(1000));
                window.request_redraw();
            },
            Event::RedrawRequested(_) => {
                // Update camera position and view-projection matrix
                let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f32();
                let time = 0.0f32;
                let camera_x = 10.0 * time.cos() as f32;
                let camera_z = 10.0 * time.sin() as f32;
                let camera_y = 10.0 ;//+ 2.0 * (time * 0.5).sin() as f32;
                let view = na::Isometry3::look_at_rh(
                    &na::Point3::new(camera_x, camera_y, camera_z),
                    &na::Point3::new(0.0, 0.0, 0.0),
                    &na::Vector3::y(),
                );
                let view_projection = projection.as_matrix() * view.to_homogeneous();
                let camera_position = na::Vector4::new(camera_x, camera_y, camera_z, 1.0);

                let draw_particles_uniforms = DrawParticlesUniforms {
                    view_projection,
                    camera_position,
                };
                queue.write_buffer(&draw_particles_uniforms_buffer, 0, bytemuck::cast_slice(&[draw_particles_uniforms]));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });

                queue.write_buffer(&pbd_uniforms_buffer, 0, bytemuck::cast_slice(&[uniforms]));
                queue.submit([]);
                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Compute Pass"),
                        ..Default::default()
                    });
                    compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                    for _ in 0..20 { 
                        compute_pass.set_pipeline(&apply_velocity_forces_pipeline);
                        compute_pass.dispatch_workgroups((num_particles as u32 + compute_config.workgroup_size - 1) / compute_config.workgroup_size, 1, 1);
                        compute_pass.set_pipeline(&voxel_constraints_pipeline);
                        compute_pass.dispatch_workgroups((num_voxels as u32 + compute_config.workgroup_size - 1) / compute_config.workgroup_size, 1, 1);
                        for partition in constraint_partitions.iter() {
                            compute_pass.set_pipeline(&apply_face_constraint_pipeline);
                            compute_pass.dispatch_workgroups(((partition.end - partition.start) + compute_config.workgroup_size - 1) / compute_config.workgroup_size, 1, 1);
                            compute_pass.set_pipeline(&step_constraint_partition_pipeline);
                            compute_pass.dispatch_workgroups(1, 1, 1);
                        }
                        compute_pass.set_pipeline(&boundary_constraints_pipeline);
                        compute_pass.dispatch_workgroups((num_particles as u32 + compute_config.workgroup_size - 1) / compute_config.workgroup_size, 1, 1);
                        compute_pass.set_pipeline(&update_velocity_pipeline);
                        compute_pass.dispatch_workgroups((num_particles as u32 + compute_config.workgroup_size - 1) / compute_config.workgroup_size, 1, 1);
                        // compute_pass.set_pipeline(&apply_damping_pipeline);
                        // compute_pass.dispatch_workgroups((num_voxels as u32 + compute_config.workgroup_size - 1) / compute_config.workgroup_size, 1, 1);
                    }

                }
                queue.submit(Some(encoder.finish()));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });
                let frame = surface
                .get_current_texture()
                .expect("Failed to acquire next swap chain texture");
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Clear pass
                {
                    let mut clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Clear Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                }

                // Particle rendering pass
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.set_bind_group(0, &render_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    render_pass.draw(0..ico_mesh.len() as u32, 0..num_particles as u32);
                }

                {
                    // let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    //     label: Some("Wireframe Pass"),
                    //     color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    //         view: &view,
                    //         resolve_target: None,
                    //         ops: wgpu::Operations {
                    //             load: wgpu::LoadOp::Load,
                    //             store: wgpu::StoreOp::Store,
                    //         },
                    //     })],
                    //     depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    //         view: &depth_view,
                    //         depth_ops: Some(wgpu::Operations {
                    //             load: wgpu::LoadOp::Load,
                    //             store: wgpu::StoreOp::Store,
                    //         }),
                    //         stencil_ops: None,
                    //     }),
                    //     ..Default::default()
                    // });
                    // render_pass.set_pipeline(&wireframe_pipeline);
                    // render_pass.set_bind_group(0, &render_bind_group, &[]);
                    // render_pass.draw(0..16, 0..flat_constraints.len() as u32);
                }

                // encoder.copy_buffer_to_buffer(
                //     &particle_buffer,
                //     0,
                //     &staging_buffer,
                //     0,
                //     (std::mem::size_of::<Particle>() * num_particles) as u64,
                // );

                encoder.copy_buffer_to_buffer(
                    &f_matrices_buffer,
                    0,
                    &f_matrices_staging_buffer,
                    0,
                    (std::mem::size_of::<FMatrices>() * (num_voxels + num_constraints)) as u64,
                );

                queue.submit(Some(encoder.finish()));
                frame.present();
                let t1 = std::time::Instant::now();
                let dt = t1.duration_since(t0).as_secs_f32();
                // println!("dt: {}", dt);

                // let buffer_slice = staging_buffer.slice(..);
                // let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                // buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                //     tx.send(result).unwrap();
                // });
                // device.poll(wgpu::Maintain::Wait);
                // if let Some(Ok(())) = pollster::block_on(rx.receive()) {
                //     let data = buffer_slice.get_mapped_range();
                //     let particles: &[Particle] = bytemuck::cast_slice(&data);
                    
                //     for (i, particle) in particles.iter().take(8).enumerate() {
                //         println!("Particle {}: position = {:?}", i, particle.x);
                //     }
                    
                //     drop(data);
                //     staging_buffer.unmap();
                // }

                if dt > 0.5 {
                    t0 = t1;
                    let f_matrices_slice = f_matrices_staging_buffer.slice(..);
                    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                    f_matrices_slice.map_async(wgpu::MapMode::Read, move |result| {
                        tx.send(result).unwrap();
                    });
                    device.poll(wgpu::Maintain::Wait);
                    if let Some(Ok(())) = pollster::block_on(rx.receive()) {
                        let data = f_matrices_slice.get_mapped_range();
                        let f_matrices: &[FMatrices] = bytemuck::cast_slice(&data);
                        
                        // Log the F matrices, Lagrange multipliers, and gradients
                        for (i, f_matrix) in f_matrices.iter().take(1).enumerate() {
                            println!("Voxel {}: F_h = [", i);
                            for row in &f_matrix.F_h {
                                println!("  {:?}", &row[..3]); // Only print the first 3 elements of each row
                            }
                            println!("]");
                            println!("Voxel {}: F_d = [", i);
                            for row in &f_matrix.F_d {
                                println!("  {:?}", &row[..3]); // Only print the first 3 elements of each row
                            }
                            println!("]");
                            // println!("Voxel {}: Lagrange multiplier (h) = {}", i, f_matrix.lagrange_h);
                            // println!("Voxel {}: Lagrange multiplier (d) = {}", i, f_matrix.lagrange_d);
                            // println!("Voxel {}: C_h = {}", i, f_matrix.C_h);
                            println!("Voxel {}: rest_positions = [", i);
                            dbg!(&f_matrix.grad_C_h);
                            // for grad in &f_matrix.grad_C_h {
                            //     println!("  {:?}", &grad[..3]); // Only print the first 3 elements of each gradient
                            // }
                            println!("]");
                            // println!("Voxel {}: C_d = {}", i, f_matrix.C_d);
                            // println!("Voxel {}: grad_C_d = [", i);
                            // for grad in &f_matrix.grad_C_d {
                            //     println!("  {:?}", &grad[..3]); // Only print the first 3 elements of each gradient
                            // }
                            // println!("]");
                        }
                        // panic!("end");
                        drop(data);
                        f_matrices_staging_buffer.unmap();
                        // std::thread::sleep(std::time::Duration::from_millis(1000));
                    }
                }
            }
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let mut builder = WindowBuilder::new();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }

    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
