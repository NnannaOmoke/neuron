use super::context::GpuContext;
use std::{ops::Deref, sync::Arc};

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

pub enum Operation {
    DotInPlace {
        a_and_out: Arc<wgpu::Buffer>,
        b: Arc<wgpu::Buffer>,
    },
    DotExtern {
        a: Arc<wgpu::Buffer>,
        b: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
    },
}

impl Operation {
    pub fn push_to_command_encoder(&self, command_encoder: &mut wgpu::CommandEncoder) {
        todo!();
    }
}
