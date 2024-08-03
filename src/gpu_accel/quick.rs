use super::shaders::init_dot_in_place;
use ndarray::{ArrayView2, ArrayViewMut2};
use std::{mem::size_of, sync::Arc};
use tokio::sync::Notify;
use wgpu::util::DeviceExt;

// TODO: Make functions gracefully handle errors.

async fn create_loaded_buffer(
    device: &wgpu::Device,
    data: &ArrayView2<'_, f32>,
    needs_cpy_src: bool,
    data_name: &str,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data.len() * size_of::<f32>()) as wgpu::BufferAddress,
        usage: if needs_cpy_src {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE
        },
        mapped_at_creation: true,
    });
    let buffer_slice = buffer.slice(..);
    let sender = Arc::new(Notify::new());
    let receiver = sender.clone();
    buffer_slice.map_async(wgpu::MapMode::Write, move |r| {
        r.expect("failed to map matmul target buffer for initialization");
        sender.notify_waiters();
    });
    device.poll(wgpu::Maintain::Poll);
    receiver.notified().await;
    if let Some(data_slice) = data.as_slice() {
        buffer_slice
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data_slice));
    } else {
        log::warn!(
            "{} cannot be cast to slice (ndarray::ArrayView::to_slice). \
            Falling back in per-float insertion. This is significantly slower.",
            data_name
        );
        let mut mapped_range = buffer_slice.get_mapped_range_mut();
        for (i, e) in data.iter().enumerate() {
            mapped_range[i..i + size_of::<f32>()].copy_from_slice(&e.to_le_bytes());
        }
    }
    buffer.unmap();
    buffer
}

async fn create_loaded_buffer_from_mut(
    device: &wgpu::Device,
    data: &ArrayViewMut2<'_, f32>,
    needs_cpy_src: bool,
    data_name: &str,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data.len() * size_of::<f32>()) as wgpu::BufferAddress,
        usage: if needs_cpy_src {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE
        },
        mapped_at_creation: true,
    });
    let buffer_slice = buffer.slice(..);
    let sender = Arc::new(Notify::new());
    let receiver = sender.clone();
    buffer_slice.map_async(wgpu::MapMode::Write, move |r| {
        r.expect("failed to map matmul target buffer for initialization");
        sender.notify_waiters();
    });
    device.poll(wgpu::Maintain::Poll);
    receiver.notified().await;
    if let Some(data_slice) = data.as_slice() {
        buffer_slice
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data_slice));
    } else {
        log::warn!(
            "{} cannot be cast to slice (ndarray::ArrayView::to_slice). \
            Falling back in per-float insertion. This is significantly slower.",
            data_name
        );
        let mut mapped_range = buffer_slice.get_mapped_range_mut();
        for (i, e) in data.iter().enumerate() {
            mapped_range[i..i + size_of::<f32>()].copy_from_slice(&e.to_le_bytes());
        }
    }
    buffer.unmap();
    buffer
}

pub async fn matmul32(mut target: ArrayViewMut2<'_, f32>, rhs: ArrayView2<'_, f32>) {
    assert_eq!(target.dim(), rhs.dim());

    let dims = target.dim();

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("could not get wgpu adapter");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("neuron quick matmul32 device"),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
            None,
        )
        .await
        .expect("could not get device from wgpu adapter");

    let target_buffer = create_loaded_buffer_from_mut(&device, &target, true, "`matmul32` `target`").await;
    let rhs_buffer = create_loaded_buffer(&device, &rhs, false, "`matmul32` `rhs`").await;
    let x_dim_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neuron matmul32 input x dim buffer"),
        contents: bytemuck::bytes_of(&dims.0),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neuron matmul32 output staging buffer"),
        size: (target.len() * size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader_module = init_dot_in_place(&device);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("neuron quick matmul32 bind group layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("neuron quick matmul32 bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: target_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: x_dim_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("neuron quick matmul32 compute pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("neuron quick matmul32 compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("neuron quick matmul32 command encoder"),
    });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("neuron quick matmul32 compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dims.0 as u32, dims.1 as u32, 0);
    }
    command_encoder.copy_buffer_to_buffer(&target_buffer, 0, &output_staging_buffer, 0, target_buffer.size());
    let submission_index = queue.submit([command_encoder.finish()]);

    let output_slice = output_staging_buffer.slice(..);

    let sender = Arc::new(Notify::new());
    let receiver = sender.clone();
    output_slice.map_async(wgpu::MapMode::Read, move |r| {
        r.expect("failed to map matmul32 output staging buffer");
        sender.notify_waiters();
    });
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index)).panic_on_timeout();
    receiver.notified().await;

    let mapped_range = output_slice.get_mapped_range();
    if let Some(mut_target_slice) = target.as_slice_mut() {
        mut_target_slice.copy_from_slice(bytemuck::cast_slice(&mapped_range));
    } else {
        for (i, e) in target.iter_mut().enumerate() {
            *e = *bytemuck::from_bytes(&mapped_range[i..i + 4]);
        }
    }
}
