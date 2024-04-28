use std::{
    borrow::Cow,
    path::Path,
    sync::{Arc, Mutex, OnceLock, Weak},
};
use thiserror::Error;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    library::VulkanLibrary,
    memory::allocator::StandardMemoryAllocator,
};

/// The global gpu context.
///
/// Prefer access and initialization with [`get_global_gpu_context`].
/// You may also prefer to initialize directly to force the usage of the default or using
/// [`GpuContext::new`].
pub static GLOBAL_GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

/// Gets the global gpu context. Returns err if it had to init and that failed.
pub fn get_global_gpu_context() -> Result<&'static GpuContext, Error> {
    if let Some(ctx) = GLOBAL_GPU_CONTEXT.get() {
        Ok(ctx)
    } else {
        let new_ctx = GpuContext::new()?;
        Ok(GLOBAL_GPU_CONTEXT.get_or_init(|| new_ctx))
    }
}

#[derive(Debug)]
pub struct GpuContext {
    device: Arc<Device>,
    queues: Vec<Arc<Queue>>,
    // TODO: Maybe look into if other allocators are better suited?
    allocator: Arc<StandardMemoryAllocator>,
}

impl GpuContext {
    pub fn init(
        device: Arc<Device>,
        queues: Vec<Arc<Queue>>,
        allocator: Arc<StandardMemoryAllocator>,
    ) -> GpuContext {
        GpuContext {
            device,
            queues,
            allocator,
        }
    }

    pub fn new() -> Result<GpuContext, Error> {
        let lib = VulkanLibrary::new()?;
        let instance = Instance::new(lib, InstanceCreateInfo::default())?;

        let physical_devices = instance.enumerate_physical_devices()?;
        if physical_devices.len() == 0 {
            return Err(Error::NoVulkanDevices);
        }

        // Looking for discrete gpu as that will likely be more powerful.
        let mut physical_device_tmp = None;
        let mut queue_family_index_tmp = None;
        for pd in physical_devices {
            let maybe_queue_family_index = pd
                .queue_family_properties()
                .iter()
                .position(|queue_family| queue_family.queue_flags.contains(QueueFlags::COMPUTE))
                .map(|i| i as u32);
            if let Some(queue_family_index) = maybe_queue_family_index {
                let device_type = pd.properties().device_type;
                if device_type == PhysicalDeviceType::DiscreteGpu {
                    physical_device_tmp = Some(pd);
                    queue_family_index_tmp = Some(queue_family_index);
                    break;
                } else if device_type as i32
                    > physical_device_tmp
                        .as_ref()
                        .map_or(-1, |fbpd: &Arc<PhysicalDevice>| {
                            fbpd.properties().device_type as i32
                        })
                {
                    physical_device_tmp = Some(pd);
                    queue_family_index_tmp = Some(queue_family_index)
                }
            }
        }

        debug_assert_eq!(
            physical_device_tmp.is_some() || queue_family_index_tmp.is_some(),
            physical_device_tmp.is_some() && queue_family_index_tmp.is_some()
        );
        let (physical_device, queue_family_index) =
            if let (Some(pd), Some(qfi)) = (physical_device_tmp, queue_family_index_tmp) {
                (pd, qfi)
            } else {
                return Err(Error::NoVulkanComputingDevices);
            };

        let (device, queues_iter) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;

        let allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Ok(GpuContext {
            device,
            queues: queues_iter.collect(),
            allocator,
        })
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    LibraryLoading(#[from] vulkano::library::LoadingError),
    #[error("no physical devices that support Vulkan also support computing")]
    NoVulkanComputingDevices,
    #[error("no physical devices that support Vulkan were found")]
    NoVulkanDevices,
    #[error(transparent)]
    VulkanValidated(#[from] vulkano::Validated<vulkano::VulkanError>),
    #[error(transparent)]
    Vulkan(#[from] vulkano::VulkanError),
}

fn device_type_prio(input: PhysicalDeviceType) -> i32 {
    match input {
        PhysicalDeviceType::Other => 0,
        PhysicalDeviceType::Cpu => 1,
        PhysicalDeviceType::VirtualGpu => 2,
        PhysicalDeviceType::IntegratedGpu => 3,
        PhysicalDeviceType::DiscreteGpu => 4,
        _ => unreachable!(),
    }
}
