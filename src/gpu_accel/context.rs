pub use wgpu::InstanceDescriptor;

use log::{debug, info, trace};
use std::{fmt::Debug, path::Path};
use super::shaders::Shaders;
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
    shaders: Shaders,
}

impl GpuContext {
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn shaders(&self) -> &Shaders {
        &self.shaders
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

        Ok(Self { device, queue, shaders: Shaders::new() })
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
