use super::context::GpuContext;
use std::{ops::Deref, sync::{Arc, OnceLock}};

pub struct ComputeContext<GpuContextPtr: Deref<Target = GpuContext>> {
    gpu_context: GpuContextPtr,
}

pub struct OperationPipeline<GpuContextPtr, ComputeContextPtr>
where
    GpuContextPtr: Deref<Target = GpuContext>,
    ComputeContextPtr: Deref<Target = ComputeContext<GpuContextPtr>>,
{
    compute_context: ComputeContextPtr,
    operations: Vec<Operation>,
}

impl<GpuContextPtr, ComputeContextPtr> OperationPipeline<GpuContextPtr, ComputeContextPtr>
where
    GpuContextPtr: Deref<Target = GpuContext>,
    ComputeContextPtr: Deref<Target = ComputeContext<GpuContextPtr>>,
{
    pub fn create_command_buffer(&self, label: Option<&str>) -> wgpu::CommandBuffer {
        self.create_command_encoder(label).finish()
    }

    /// Creates an unfinished [`wgpu::CommandEncoder`] to which additional operations
    /// or commands can be added.
    ///
    /// You may be looking for [`OperationPipeline::create_command_buffer`].
    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        let mut command_encoder = self
            .compute_context
            .gpu_context
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label });
        for operation in &self.operations {
            operation.push_to_command_encoder(&mut command_encoder);
        }

        command_encoder
    }

    pub fn push_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }
}

pub struct WgpuPipelines {
    dot_in_place: OnceLock<wgpu::ComputePipeline>,
    dot_extern: OnceLock<wgpu::ComputePipeline>,
}

pub enum Operation {
    CopyBufferToBuffer {
        source: Arc<wgpu::Buffer>,
        source_offset: wgpu::BufferAddress,
        target: Arc<wgpu::Buffer>,
        target_offset: wgpu::BufferAddress,
        copy_len: wgpu::BufferAddress,
    },
    DotInPlace {
        a_and_out: Arc<wgpu::Buffer>,
        b: Arc<wgpu::Buffer>,
        wgpu_label: String,
    },
    DotExtern {
        a: Arc<wgpu::Buffer>,
        b: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
        wgpu_label: String,
    },
}

impl Operation {
    pub fn push_to_command_encoder(&self, command_encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::CopyBufferToBuffer {
                source,
                source_offset,
                target,
                target_offset,
                copy_len,
            } => {
                command_encoder.copy_buffer_to_buffer(
                    &source,
                    *source_offset,
                    &target,
                    *target_offset,
                    *copy_len,
                );
            }
            Self::DotInPlace {
                a_and_out,
                b,
                wgpu_label,
            } => {
                let mut ce = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&wgpu_label),
                    timestamp_writes: None,
                });
                todo!();
            }
            Self::DotExtern {
                a,
                b,
                output,
                wgpu_label,
            } => {
                let mut ce = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&wgpu_label),
                    timestamp_writes: None,
                });
                todo!();
            }
        }
    }
}
