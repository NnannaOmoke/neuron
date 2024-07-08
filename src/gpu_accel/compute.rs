use super::{context::GpuContext, shaders::{Shaders, SHADER_MAIN_NAME}};
use std::{
    ops::Deref,
    sync::{Arc, OnceLock},
};

pub struct ComputeContext<GpuContextPtr: Deref<Target = GpuContext>> {
    gpu_context: GpuContextPtr,
    pipelines: WgpuPipelines,
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

impl WgpuPipelines {
    pub fn init(&self, gpu_context: &GpuContext, shaders: &Shaders) {
        const BUFFER_BIND_GROUP_LAYOUT_ENTRY_DEFAULT: wgpu::BindGroupLayoutEntry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let dot_in_place_bind_group_layout = gpu_context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Neuron dot-in-place binding layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    ..BUFFER_BIND_GROUP_LAYOUT_ENTRY_DEFAULT
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    ..BUFFER_BIND_GROUP_LAYOUT_ENTRY_DEFAULT
                },
            ],
        });
        let dot_in_place_layout = gpu_context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Neuron dot-in-place layout"),
            bind_group_layouts: &[&dot_in_place_bind_group_layout],
            push_constant_ranges: &[],
        });
        let dot_in_place_pipeline = gpu_context.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Neuron dot-in-place pipeline"),
            layout: Some(&dot_in_place_layout),
            module: shaders.get_dot_in_place(gpu_context.device()),
            entry_point: &SHADER_MAIN_NAME,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        self.dot_in_place.set(dot_in_place_pipeline).ok();

        let dot_extern_bind_group_layout = gpu_context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Neuron dot-in-place binding layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    ..BUFFER_BIND_GROUP_LAYOUT_ENTRY_DEFAULT
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    ..BUFFER_BIND_GROUP_LAYOUT_ENTRY_DEFAULT
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    ..BUFFER_BIND_GROUP_LAYOUT_ENTRY_DEFAULT
                }
            ],
        });
        let dot_extern_layout = gpu_context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Neuron dot-in-place layout"),
            bind_group_layouts: &[&dot_extern_bind_group_layout],
            push_constant_ranges: &[],
        });
        let dot_extern_pipeline = gpu_context.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Neuron dot-extern pipeline"),
            layout: Some(&dot_extern_layout),
            module: shaders.get_dot_in_place(gpu_context.device()),
            entry_point: &SHADER_MAIN_NAME,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        self.dot_extern.set(dot_extern_pipeline).ok();
    }

    pub fn new(gpu_context: &GpuContext, shaders: &Shaders) -> Self {
        let wgpu_pipelines = Self::new_uninit();
        wgpu_pipelines.init(gpu_context, shaders);
        wgpu_pipelines
    }
    
    pub fn new_uninit() -> Self {
        Self {
            dot_in_place: OnceLock::new(),
            dot_extern: OnceLock::new(),
        }
    }
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
