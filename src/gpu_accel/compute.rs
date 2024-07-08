use super::context::GpuContext;
use std::{
    ops::Deref,
    sync::{Arc, OnceLock},
};

pub struct ComputeContext<GpuContextPtr: Deref<Target = GpuContext>> {
    gpu_context: GpuContextPtr,
}

impl<GpuContextPtr: Deref<Target = GpuContext>> ComputeContext<GpuContextPtr> {
    pub fn create_pipeline(&self) -> OperationPipeline<GpuContextPtr, &Self> {
        OperationPipeline::new(self)
    }

    pub fn create_pipeline_arc(self: Arc<Self>) -> OperationPipeline<GpuContextPtr, Arc<Self>> {
        OperationPipeline::new(self)
    }
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

    pub fn new(compute_context: ComputeContextPtr) -> Self {
        Self {
            compute_context,
            operations: Vec::new(),
        }
    }

    pub fn new_with_operations(compute_context: ComputeContextPtr, operations: Vec<Operation>) -> Self {
        Self {
            compute_context,
            operations,
        }
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
        wgpu_label: Option<String>,
    },
    DotExtern {
        a: Arc<wgpu::Buffer>,
        b: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
        wgpu_label: Option<String>,
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
                    label: wgpu_label.as_ref().map(|s| s.as_str()),
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
                    label: wgpu_label.as_ref().map(|s| s.as_str()),
                    timestamp_writes: None,
                });
                todo!();
            }
        }
    }
}
