pub use wgpu::InstanceDescriptor;

use super::shaders::{SHADER_MAIN_NAME, Shaders};
use log::{debug, info, trace};
use std::{fmt::Debug, path::Path, sync::OnceLock};
use thiserror::Error;

pub struct DeviceOptions<'a, F: DeviceSelectorFn> {
    pub selector: DeviceSelector<F>,
    pub limits: wgpu::Limits,
    pub wgpu_label: Option<&'a str>,
    pub trace_path: Option<&'a Path>,
}

impl Default for DeviceOptions<'static, DeviceSelectorFDummy> {
    fn default() -> Self {
        Self {
            selector: DeviceSelector::HighPerformance,
            limits: wgpu::Limits::downlevel_defaults(),
            wgpu_label: Some("neuron selected compute device"),
            trace_path: None,
        }
    }
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    wgpu_pipelines: WgpuPipelines,
}

impl GpuContext {
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn wgpu_pipelines(&self) -> &WgpuPipelines {
        &self.wgpu_pipelines
    }

    pub async fn new<'a, F: DeviceSelectorFn>(
        instance_descriptor: InstanceDescriptor,
        device_options: DeviceOptions<'a, F>,
    ) -> Result<Self, Error> {
        let backends = instance_descriptor.backends;
        let instance = wgpu::Instance::new(instance_descriptor);

        let adapter = match device_options.selector {
            DeviceSelector::HighPerformance => {
                instance
                    .request_adapter(&wgpu::RequestAdapterOptionsBase {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                    })
                    .await
            }
            DeviceSelector::LowPower => {
                instance
                    .request_adapter(&wgpu::RequestAdapterOptionsBase {
                        power_preference: wgpu::PowerPreference::LowPower,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                    })
                    .await
            }
            DeviceSelector::Custom(f) => {
                let adapters = instance.enumerate_adapters(backends);
                if adapters.is_empty() {
                    return Err(Error::NoAdapter);
                }
                Some(f(adapters))
            }
        }
        .ok_or(Error::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: device_options.wgpu_label,
                    required_features: wgpu::Features::default(),
                    required_limits: device_options.limits,
                },
                device_options.trace_path,
            )
            .await?;

        let shaders = Shaders::new();
        let wgpu_pipelines = WgpuPipelines::new(&device, &shaders);

        Ok(Self {
            device,
            queue,
            wgpu_pipelines,
        })
    }
}

pub struct WgpuPipelines {
    dot_in_place: OnceLock<wgpu::ComputePipeline>,
    dot_extern: OnceLock<wgpu::ComputePipeline>,
}

impl WgpuPipelines {
    pub fn init(&self, device: &wgpu::Device, shaders: &Shaders) {
        const BUFFER_BIND_GROUP_LAYOUT_ENTRY_DEFAULT: wgpu::BindGroupLayoutEntry =
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let dot_in_place_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let dot_in_place_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Neuron dot-in-place layout"),
            bind_group_layouts: &[&dot_in_place_bind_group_layout],
            push_constant_ranges: &[],
        });
        let dot_in_place_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Neuron dot-in-place pipeline"),
                layout: Some(&dot_in_place_layout),
                module: shaders.get_dot_in_place(device),
                entry_point: &SHADER_MAIN_NAME,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        self.dot_in_place.set(dot_in_place_pipeline).ok();

        let dot_extern_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    },
                ],
            });
        let dot_extern_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Neuron dot-in-place layout"),
            bind_group_layouts: &[&dot_extern_bind_group_layout],
            push_constant_ranges: &[],
        });
        let dot_extern_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Neuron dot-extern pipeline"),
                layout: Some(&dot_extern_layout),
                module: shaders.get_dot_in_place(device),
                entry_point: &SHADER_MAIN_NAME,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        self.dot_extern.set(dot_extern_pipeline).ok();
    }

    pub fn new(device: &wgpu::Device, shaders: &Shaders) -> Self {
        let wgpu_pipelines = Self::new_uninit();
        wgpu_pipelines.init(device, shaders);
        wgpu_pipelines
    }

    pub fn new_uninit() -> Self {
        Self {
            dot_in_place: OnceLock::new(),
            dot_extern: OnceLock::new(),
        }
    }
}

#[derive(Default)]
pub enum DeviceSelector<F: DeviceSelectorFn> {
    #[default]
    HighPerformance,
    LowPower,
    Custom(F),
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("could not get wgpu adapter")]
    NoAdapter,
    #[error("could not get wgpu device from adapter")]
    NoDevice { used_adapter: wgpu::Adapter },
    #[error(transparent)]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

pub trait DeviceSelectorFn: FnOnce(Vec<wgpu::Adapter>) -> wgpu::Adapter {}

impl<T: FnOnce(Vec<wgpu::Adapter>) -> wgpu::Adapter> DeviceSelectorFn for T {}

pub type DeviceSelectorFDummy = fn(Vec<wgpu::Adapter>) -> wgpu::Adapter;

#[cfg(test)]
mod tests {
    use super::*;

    async fn gpu_context_new_inner() {
        for device_options in [
            DeviceOptions {
                selector: DeviceSelector::HighPerformance,
                ..Default::default()
            },
            DeviceOptions {
                selector: DeviceSelector::LowPower,
                ..Default::default()
            },
            DeviceOptions {
                selector: DeviceSelector::Custom(|mut adapters| adapters.remove(0)),
                ..Default::default()
            },
        ] {
            GpuContext::new(InstanceDescriptor::default(), device_options)
                .await
                .unwrap();
        }
    }

    #[test]
    fn gpu_context_new() {
        async fn inner() {
            for device_options in [
                DeviceOptions {
                    selector: DeviceSelector::HighPerformance,
                    ..Default::default()
                },
                DeviceOptions {
                    selector: DeviceSelector::LowPower,
                    ..Default::default()
                },
                DeviceOptions {
                    selector: DeviceSelector::Custom(|mut adapters| adapters.remove(0)),
                    ..Default::default()
                },
            ] {
                GpuContext::new(InstanceDescriptor::default(), device_options)
                    .await
                    .unwrap();
            }
        }

        pollster::block_on(inner())
    }
}
