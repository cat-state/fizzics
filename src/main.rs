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


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    x: na::Vector3<f32>,
    mass: f32,
    v: na::Vector3<f32>,
    _padding: f32
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
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FaceConstraint {
   l: [i32; 4],
   r: [i32; 4]
}

fn voxel_cube(size: u32) -> (Vec<Voxel>, (Vec<FaceConstraint>, Vec<FaceConstraint>, Vec<FaceConstraint>)) {
    /*
     *     7 ---- 6
     *    ..     .|
     *   . .    . |
     *  3 ---- 2  |
     *  . \.   |  |
     *  .  4 --|- 5
     *  . .    |.
     *  0 ---- 1
     */
    // let x_faces = [[1, 2, 3, 2], [5, 4, 7, 6]];
    // let y_faces = [[3, 2, 1, 0], [7, 6, 5, 4]];
    // let z_faces = [[4, 5, 6, 7], [0, 1, 2, 3]];
    let offsets = [
        na::Vector3::new(-1.0, -1.0, -1.0) * 0.5,
        na::Vector3::new(1.0, -1.0, -1.0) * 0.5,
        na::Vector3::new(1.0, 1.0, -1.0) * 0.5,
        na::Vector3::new(-1.0, 1.0, -1.0) * 0.5,
        na::Vector3::new(-1.0, -1.0, 1.0) * 0.5,
        na::Vector3::new(1.0, -1.0, 1.0) * 0.5,
        na::Vector3::new(1.0, 1.0, 1.0) * 0.5,
        na::Vector3::new(-1.0, 1.0, 1.0) * 0.5,
    ];
    let zero = na::Vector3::zeros();
    let center = na::Vector3::new(0.5 * size as f32, 0.5 * size as f32, 0.5 * size as f32);

    let mut voxels = Vec::new();
    let mut x_constraints = Vec::new();
    let mut y_constraints = Vec::new();
    let mut z_constraints = Vec::new();
    for x in 0..size {
        for y in 0..size {
            for z in 0..size {
                let p = 2.0 * na::Vector3::new(x as f32, y as f32, z as f32);
                let voxel = Voxel {
                    particles: offsets.map(|offset| 
                        Particle { 
                            x: p + offset - center, 
                            mass: 1.0,
                            v: zero,
                            _padding: 0.0,
                        }
                    )
                };
                voxels.push(voxel);
                let midx = x * size * size + y * size + z;
                let mpidx = (midx * 8) as i32;
                if x != 0 {    
                    let vidx = (x - 1) * size * size + y * size + z;
                    let vpidx = (vidx * 8) as i32;
                    let c = FaceConstraint {
                        l: [vpidx + 1, vpidx + 5, vpidx + 6, vpidx + 2],
                        r: [mpidx + 0, mpidx + 4, mpidx + 7, mpidx + 3]
                    };
                    x_constraints.push(c);
                }
                if x != (size - 1) {
                    let vidx = (x + 1) * size * size + y * size + z;
                    let vpidx = (vidx * 8) as i32;
                    let c = FaceConstraint {
                        l: [mpidx + 1, mpidx + 5, mpidx + 6, mpidx + 2],
                        r: [vpidx + 0, vpidx + 4, vpidx + 7, vpidx + 3]
                    };
                    x_constraints.push(c);
                }
                if y != 0 {
                    let vidx = x * size * size + (y - 1) * size + z;
                    let vpidx = (vidx * 8) as i32;
                    let c = FaceConstraint {
                        l: [mpidx + 0, mpidx + 4, mpidx + 5, mpidx + 1],
                        r: [-(vpidx + 3), -(vpidx + 7), -(vpidx + 6), -(vpidx + 2)]
                    };
                    y_constraints.push(c);
                }
                if y != (size - 1) {
                    let vidx = x * size * size + (y + 1) * size + z;
                    let vpidx = (vidx * 8) as i32;
                    let c = FaceConstraint {
                        l: [vpidx + 0, vpidx + 4, vpidx + 5, vpidx + 1],
                        r: [-(mpidx + 3), -(mpidx + 7), -(mpidx + 6), -(mpidx + 2)]
                    };
                    y_constraints.push(c);
                }
                if z != 0 {
                    let vidx = x * size * size + y * size + (z - 1);
                    let vpidx = (vidx * 8) as i32;
                    let c = FaceConstraint {
                        l: [mpidx + 0, mpidx + 1, mpidx + 2, mpidx + 3],
                        r: [-(vpidx + 4), -(vpidx + 5), -(vpidx + 6), -(vpidx + 7)]
                    };
                    z_constraints.push(c);
                }
                if z != (size - 1) {
                    let vidx = x * size * size + y * size + (z + 1);
                    let vpidx = (vidx * 8) as i32;
                    let c = FaceConstraint {
                        l: [vpidx + 0, vpidx + 1, vpidx + 2, vpidx + 3],
                        r: [-(mpidx + 4), -(mpidx + 5), -(mpidx + 6), -(mpidx + 7)]
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
struct PBDUniforms {
    i_mouse: na::Vector4<f32>,
    i_resolution: na::Vector2<f32>,
    i_frame: i32,
    constraint_phase: u32,
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
                required_features: wgpu::Features::empty().union(wgpu::Features::POLYGON_MODE_LINE),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let voxel_constraints = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Voxel Constraints Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pbd.wgsl").into()),
    });

    let draw_particles_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Draw Particles Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/draw_particles.wgsl").into()),
    });

    let ico_mesh = icosahedron_sphere(0.4, 1);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sphere Mesh Vertex Buffer"),
        contents: bytemuck::cast_slice(&ico_mesh),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

    let voxel_constraints_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Voxel Constraints Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &voxel_constraints,
        entry_point: "voxel_constraint",
        compilation_options: Default::default(),
        cache: None,
    });

    let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Collision Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &voxel_constraints,
        entry_point: "handle_particle_collisions",
        compilation_options: Default::default(),
        cache: None,
    });

    let x_face_constraints_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Face Constraints Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &voxel_constraints,
        entry_point: "apply_x_face_constraints",
        compilation_options: Default::default(),
        cache: None,
    });

    let y_face_constraints_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Face Constraints Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &voxel_constraints,
        entry_point: "apply_y_face_constraints",
        compilation_options: Default::default(),
        cache: None,
    });
    

    let z_face_constraints_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Face Constraints Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &voxel_constraints,
        entry_point: "apply_z_face_constraints",
        compilation_options: Default::default(),
        cache: None,
    });

    let (voxels, xyz_constraints) = voxel_cube(4);
    let flat_constraints = xyz_constraints.0.into_iter().chain(xyz_constraints.1.into_iter()).chain(xyz_constraints.2.into_iter()).collect::<Vec<FaceConstraint>>();
    let num_voxels = voxels.len();
    let num_particles = num_voxels * 8;
    dbg!( bytemuck::cast_slice::<Voxel, u8>(&voxels).len());
    let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particles Buffer"),
        contents: bytemuck::cast_slice(&voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let constraint_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Constraints Buffer"),
        contents: bytemuck::cast_slice(&flat_constraints),
        usage: wgpu::BufferUsages::STORAGE,
    });
    

    let uniforms = PBDUniforms {
        i_mouse: na::Vector4::<f32>::new(0.0, 0.0, 0.0, 0.0),
        i_resolution: na::Vector2::<f32>::new(size.width as f32, size.height as f32),
        i_frame: 0,
        constraint_phase: 0,
    };

    let pbd_uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PBD Uniforms Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                resource: pbd_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: constraint_buffer.as_entire_binding(),
            },
        ],
    });

    let aspect_ratio = size.width as f32 / size.height as f32;
    let projection = na::Perspective3::new(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 100.0);
    let view = na::Isometry3::look_at_rh(
        &na::Point3::new(5.0, 5.0, 5.0),
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
        ],
    });

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
                window.request_redraw();
            },
            Event::RedrawRequested(_) => {
                // Update camera position and view-projection matrix
                let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
                let time = 0.0f64;
                let camera_x = 16.0 * time.cos() as f32;
                let camera_z = 16.0 * time.sin() as f32;
                let camera_y = 15.0 ;//+ 2.0 * (time * 0.5).sin() as f32;
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

                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Compute Pass"),
                        ..Default::default()
                    });
                    compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                    // compute_pass.set_pipeline(&collision_pipeline);
                    // compute_pass.dispatch_workgroups(num_particles as u32, 1, 1);
                    for _ in 0..4 { 
                        compute_pass.set_pipeline(&voxel_constraints_pipeline);
                        compute_pass.dispatch_workgroups(num_voxels as u32, 1, 1);     
                        compute_pass.set_pipeline(&x_face_constraints_pipeline);
                        compute_pass.dispatch_workgroups(flat_constraints.len() as u32 / 3u32, 1, 1);
                        compute_pass.set_pipeline(&y_face_constraints_pipeline);
                        compute_pass.dispatch_workgroups(flat_constraints.len() as u32 / 3u32, 1, 1);
                        compute_pass.set_pipeline(&z_face_constraints_pipeline);
                        compute_pass.dispatch_workgroups(flat_constraints.len() as u32 / 3u32, 1, 1);


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

                queue.submit(Some(encoder.finish()));
                frame.present();
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
