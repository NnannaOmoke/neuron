use pollster::block_on;
use std::{
    path::Path,
    sync::{Mutex, OnceLock},
};

pub static GLOBAL_GPU_OPTIONS: Mutex<Option<GpuContextOptions<'static>>> = Mutex::new(None);
/// The global gpu context.
///
/// Prefer access and initialization with [`get_global_gpu_context`].
/// You may also prefer to initialize directly to force the usage of the default or using
/// [`GpuContext::new`].
pub static GLOBAL_GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

/// Gets the global gpu context. Initializes using the options in [`GLOBAL_GPU_OPTIONS`]
/// or default if not initialized.
pub fn get_global_gpu_context() -> &'static GpuContext {
    GLOBAL_GPU_CONTEXT.get_or_init(|| {
        if let Ok(gpu_options_mutex) = GLOBAL_GPU_OPTIONS.lock() {
            if let Some(gpu_options) = gpu_options_mutex.as_ref() {
                GpuContext::new(gpu_options)
            } else {
                GpuContext::default()
            }
        } else {
            GpuContext::default()
        }
    })
}

#[derive(Debug)]
pub struct GpuContext {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn default_async() -> GpuContext {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
        GpuContext::init(instance, device, queue)
    }

    pub fn init(instance: wgpu::Instance, device: wgpu::Device, queue: wgpu::Queue) -> GpuContext {
        GpuContext {
            instance,
            device,
            queue,
        }
    }

    pub fn new(options: &GpuContextOptions) -> GpuContext {
        block_on(GpuContext::new_async(options))
    }

    pub async fn new_async<'a>(options: &GpuContextOptions<'a>) -> GpuContext {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference,
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: options.device_label,
                    ..Default::default()
                },
                options.device_trace_path,
            )
            .await
            .unwrap();
        GpuContext::init(instance, device, queue)
    }
}

impl Default for GpuContext {
    /// Easiest way of getting a gpu context.
    ///
    /// Blocks for wgpu futures and panics on any error in the creation process.
    fn default() -> Self {
        block_on(GpuContext::default_async())
    }
}

#[derive(Default)]
pub struct GpuContextOptions<'a> {
    power_preference: wgpu::PowerPreference,
    device_label: Option<&'a str>,
    device_trace_path: Option<&'a Path>,
}

pub mod wgpu_reexports {
    pub use wgpu::PowerPreference;
}
