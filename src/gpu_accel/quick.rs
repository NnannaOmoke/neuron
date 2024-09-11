//! Todo: `matmul32_in_place_rhs`?

use crate::gpu_accel::utils::{create_loaded_buffer, create_loaded_buffer_from_mut};

use super::{shaders::init_matmul, utils};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use std::{mem::size_of, sync::Arc};
use thiserror::Error;
use tokio::sync::Notify;
use wgpu::util::DeviceExt;

async fn get_wgpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("could not get wgpu adapter");
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("neuron quick matmul32 device"),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
            None,
        )
        .await
        .expect("could not get device from wgpu adapter")
}

fn matmul32_create_bind_group_and_pipeline(
    device: &wgpu::Device,
    lhs_buf: &wgpu::Buffer,
    rhs_buf: &wgpu::Buffer,
    dims_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::ComputePipeline) {
    let shader_module = init_matmul(&device);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("neuron quick matmul32 bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("neuron quick matmul32 bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buf.as_entire_binding(),
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

    (bind_group, compute_pipeline)
}

/// Preforms matrix multiplication of f32s into a mutable output array view.
pub async fn matmul32_extern(
    lhs: ArrayView2<'_, f32>,
    rhs: ArrayView2<'_, f32>,
    mut out: ArrayViewMut2<'_, f32>,
) -> Result<(), utils::Error> {
    assert_eq!(
        lhs.dim().1,
        rhs.dim().0,
        "col count of lhs must == row count of rhs"
    );
    let out_dim = out.dim();
    assert_eq!(
        out_dim,
        (lhs.dim().0, rhs.dim().1),
        "out dims must be row_count(lhs) * col_count(rhs)"
    );

    let (device, queue) = get_wgpu_device().await;

    // See docs for `utils::create_loaded_buffer`.
    let lhs_buffer = create_loaded_buffer(&device, &lhs, "`matmul32_extern` `lhs`").await?;
    let rhs_buffer = create_loaded_buffer(&device, &rhs, "`matmul33_extern` `rhs`").await?;
    let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neuron matmul32 input dims buffer"),
        contents: bytemuck::bytes_of(&[lhs.dim().0, lhs.dim().1, rhs.dim().0, rhs.dim().1]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output_buffer_size = (out_dim.0 * out_dim.1 * size_of::<f32>()) as wgpu::BufferAddress;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neuron matmul32 output buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neuron matmul32 output staging buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let (bind_group, compute_pipeline) = matmul32_create_bind_group_and_pipeline(
        &device,
        &lhs_buffer,
        &rhs_buffer,
        &dims_buffer,
        &output_buffer,
    );

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("neuron quick matmul32_extern command encoder"),
    });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("neuron quick matmul32_extern compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(out_dim.0 as u32, out_dim.1 as u32, 0);
    }
    command_encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &output_staging_buffer,
        0,
        output_buffer_size as wgpu::BufferAddress,
    );
    let submission_index = queue.submit([command_encoder.finish()]);

    let output_slice = output_staging_buffer.slice(..);

    let (sender, receiver) = tokio::sync::oneshot::channel();
    output_slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r)
            .expect("failed to send buffer mapping result through channel")
    });
    device
        .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index))
        .panic_on_timeout();
    receiver.await??;

    let mapped_range = output_slice.get_mapped_range();
    if let Some(mut_target_slice) = out.as_slice_mut() {
        mut_target_slice.copy_from_slice(bytemuck::cast_slice(&mapped_range));
    } else {
        log::warn!(
            "{} cannot be cast to slice (ndarray::ArrayView::to_slice_mut). \
            Falling back in per-float insertion. This is significantly slower.",
            out
        );
        for (i, e) in out.iter_mut().enumerate() {
            *e = *bytemuck::from_bytes(&mapped_range[i..i + 4]);
        }
    }

    Ok(())
}

/// Same as matmul32_extern but it outputs into the lhs CPU buffer.
///
/// Note that this has the same GPU memory footprint, it just has the benefit of
/// sometimes letting you get away with skipping an extra cpu allocation, especially
/// with same-dim matrices.
pub async fn matmul32_in_place_lhs(
    mut lhs: ArrayViewMut2<'_, f32>,
    rhs: ArrayView2<'_, f32>,
) -> Result<(), utils::Error> {
    let lhs_dim = lhs.dim();
    let rhs_dim = rhs.dim();
    assert_eq!(lhs_dim.1, rhs_dim.0, "lhs.cols must == rhs.rows");
    assert_eq!(lhs_dim.1, rhs_dim.1, "lhs.cols must == rhs.cols");

    let (device, queue) = get_wgpu_device().await;

    // See docs for `utils::create_loaded_buffer`.
    let lhs_buf =
        create_loaded_buffer_from_mut(&device, &lhs, "`matmul32_in_place_lhs` `lhs`").await?;
    let rhs_buf = create_loaded_buffer(&device, &rhs, "`matmul33_in_place_lhs` `rhs`").await?;
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neuron matmul32 input dims buffer"),
        contents: bytemuck::bytes_of(&[lhs_dim.0, lhs_dim.1, rhs_dim.0, rhs_dim.1]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output_buffer_size = (lhs_dim.0 * lhs_dim.1 * size_of::<f32>()) as wgpu::BufferAddress;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neuron matmul32 output buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neuron matmul32 output staging buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let (bind_group, compute_pipeline) = matmul32_create_bind_group_and_pipeline(
        &device,
        &lhs_buf,
        &rhs_buf,
        &dims_buf,
        &output_buf,
    );

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("neuron quick matmul32_extern command encoder"),
    });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("neuron quick matmul32_extern compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(lhs_dim.0 as u32, lhs_dim.1 as u32, 0);
    }
    command_encoder.copy_buffer_to_buffer(
        &output_buf,
        0,
        &output_staging_buf,
        0,
        output_buffer_size as wgpu::BufferAddress,
    );
    let submission_index = queue.submit([command_encoder.finish()]);

    let output_slice = output_staging_buf.slice(..);

    let (sender, receiver) = tokio::sync::oneshot::channel();
    output_slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r)
            .expect("failed to send buffer mapping result through channel")
    });
    device
        .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index))
        .panic_on_timeout();
    receiver.await??;

    let mapped_range = output_slice.get_mapped_range();
    if let Some(mut_target_slice) = lhs.as_slice_mut() {
        mut_target_slice.copy_from_slice(bytemuck::cast_slice(&mapped_range));
    } else {
        log::warn!(
            "{} cannot be cast to slice (ndarray::ArrayView::to_slice_mut). \
            Falling back in per-float insertion. This is significantly slower.",
            lhs
        );
        for (i, e) in lhs.iter_mut().enumerate() {
            *e = *bytemuck::from_bytes(&mapped_range[i..i + 4]);
        }
    }

    Ok(())
}
