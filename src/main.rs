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
    v: na::Vector3<f32>,
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Voxel {
    particles: [Particle; 8],
}

fn voxel_cube(size: u32) -> Vec<Voxel> {
    let offsets = na::Vector3::<f32>::new(1.0, 0.0, 0.0);
    let zero = na::Vector3::<f32>::zeros();
    (0..size*size*size).map(|i| {
        let x = i % size;
        let y = (i / size) % size;
        let z = i / (size * size);
        na::Vector3::<f32>::new(x as f32, y as f32, z as f32)
    }).map(|p|
        Voxel {
            particles: [
                Particle { x:p, v: zero },
                Particle { x: p + offsets.xyy(), v: zero },
                Particle { x: p + offsets.yxy(), v: zero },
                Particle { x: p + offsets.xxy(), v: zero },
                Particle { x: p + offsets.yyx(), v: zero },
                Particle { x: p + offsets.xyx(), v: zero },
                Particle { x: p + offsets.yxx(), v: zero },
                Particle { x: p + offsets.xxx(), v: zero },
            ]
        }
    )
    .map(|mut v| {
        v.particles.iter_mut().for_each(|p| {
            p.x = p.x - na::Vector3::<f32>::new(size as f32 / 2.0, size as f32 / 2.0, size as f32 / 2.0);
        });
        v
    })
    .collect::<Vec<_>>()
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

fn sphere_mesh(radius: f32, subdivisions: usize) -> Vec<MeshVertex> {
    let mut square = vec![
        na::Vector3::new(-1.0f32, 1.0, 1.0),
        na::Vector3::new(1.0, 1.0, 1.0),
        na::Vector3::new(1.0, -1.0, 1.0),
        na::Vector3::new(-1.0, 1.0, 1.0),
    ];
    for i in 0..subdivisions {
        let mut new_square = Vec::new();
        for subsquares in square.chunks(4) {
            let (tl, tr, br, bl) = (subsquares[0], subsquares[1], subsquares[2], subsquares[3]);
            let center = (tl + tr + br + bl) / 4.0f32;
            let ct = (tl + tr) / 2.0f32;
            let cl = (tl + bl) / 2.0f32;
            let cr = (tr + br) / 2.0f32;
            let cb = (bl + br) / 2.0f32;
            new_square.push(tl);
            new_square.push(ct);
            new_square.push(center);
            new_square.push(cl);

            new_square.push(ct);
            new_square.push(tr);
            new_square.push(cr);
            new_square.push(center);

            new_square.push(center);
            new_square.push(cr);
            new_square.push(br);
            new_square.push(cb);

            new_square.push(cl);
            new_square.push(center);
            new_square.push(cb);
            new_square.push(bl);
        }
        square = new_square;
    }

    let mut triangulated_square = Vec::new();
    for s in square.chunks(4) {
        let (tl, tr, br, bl) = (s[0], s[1], s[2], s[3]);
        triangulated_square.extend([bl, tr, tl, tr, bl, br]);
    }

    // Create a cube using the triangulated square as a face with rotations
    let mut cube_vertices = Vec::new();
    let rotations = [
        na::Rotation3::identity(),
        // na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), std::f32::consts::PI),
        na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), std::f32::consts::PI),
        // na::Rotation3::from_axis_angle(&na::Vector3::y_axis(), -std::f32::consts::FRAC_PI_2),
        // na::Rotation3::from_axis_angle(&na::Vector3::x_axis(), -std::f32::consts::FRAC_PI_2),
        // na::Rotation3::from_axis_angle(&na::Vector3::x_axis(), std::f32::consts::FRAC_PI_2),
    ];

    for rotation in rotations.iter() {
        cube_vertices.extend(triangulated_square.iter().map(|v| rotation * v));
    }


    cube_vertices.iter()
    .map(|v| MeshVertex { position: (v).normalize() * radius, normal: (v).normalize() }).collect()
}

fn normalize(v: na::Vector3<f32>, scale: f32) -> na::Vector3<f32> {
    v.normalize() * scale
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PBDUniforms {
    i_mouse: na::Vector4<f32>,
    i_resolution: na::Vector2<f32>,
    i_frame: i32,
    _padding: [u8; 4]
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
                required_features: wgpu::Features::empty(),
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

    let sphere_mesh = sphere_mesh(0.5, 4);

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sphere Mesh Vertex Buffer"),
        contents: bytemuck::cast_slice(&sphere_mesh),
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

    let voxels = voxel_cube(3);
    let num_voxels = voxels.len();
    let num_particles = num_voxels * 8;
    dbg!( bytemuck::cast_slice::<Voxel, u8>(&voxels).len());
    let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particles Buffer"),
        contents: bytemuck::cast_slice(&voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    

    let uniforms = PBDUniforms {
        i_mouse: na::Vector4::<f32>::new(0.0, 0.0, 0.0, 0.0),
        i_resolution: na::Vector2::<f32>::new(size.width as f32, size.height as f32),
        i_frame: 0,
        _padding: [0; 4],
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

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
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

    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                surface.configure(&device, &config);
                window.request_redraw();
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            },
            Event::RedrawRequested(_) => {
                // Update camera position and view-projection matrix
                let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f32();
                let camera_x = 10.0 * time.cos();
                let camera_z = 10.0 * time.sin();
                let camera_y = 5.0 + 2.0 * (time * 0.5).sin();
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
                    compute_pass.set_pipeline(&voxel_constraints_pipeline);
                    compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                    compute_pass.dispatch_workgroups(num_voxels as u32, 1, 1);

                    compute_pass.set_pipeline(&collision_pipeline);
                    compute_pass.dispatch_workgroups(num_particles as u32, 1, 1);
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
                    render_pass.draw(0..sphere_mesh.len() as u32, 0..num_particles as u32);
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
