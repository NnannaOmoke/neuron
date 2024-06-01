use log::{debug, error, info, trace, warn};
#[cfg(not(feature = "async_impls"))]
use std::sync::Mutex;
use std::{borrow::Cow, fmt::Debug, path::Path, sync::Arc};
use thiserror::Error;
#[cfg(feature = "async_impls")]
use tokio::sync::Mutex;
use vulkano::{
    buffer::Buffer,
    command_buffer::allocator::{
        StandardCommandBufferAlloc, StandardCommandBufferAllocator,
        StandardCommandBufferAllocatorCreateInfo,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    library::VulkanLibrary,
    memory::allocator::StandardMemoryAllocator,
};

pub struct GpuContext {
    device: Arc<Device>,
    queue_family_index: u32,
    queues: Box<[Arc<Queue>]>,
    next_queue: Arc<Mutex<usize>>,
    memory_allocator: StandardMemoryAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,
    // Likely going to change this method of storage or remove it.
    buffers: Vec<Arc<Buffer>>,
}

impl GpuContext {
    pub fn new(
        device_preference: DeviceSelector<
            impl FnOnce(&mut dyn Iterator<Item = Arc<PhysicalDevice>>) -> (Arc<PhysicalDevice>, u32),
        >,
        queue_family_preference: QueueFamilySelector,
    ) -> Result<GpuContext, Error> {
        let library = VulkanLibrary::new()?;
        let instance = Instance::new(library, InstanceCreateInfo::default())?;
        trace!("Created Vulkan library and instance.");

        let vulkan_devices = instance.enumerate_physical_devices()?;
        if vulkan_devices.len() == 0 {
            return Err(Error::NoVulkanDevices);
        }
        let (physical_device, queue_family_index) = select_device_and_queue_family(
            vulkan_devices,
            device_preference,
            queue_family_preference,
        )?;
        info!("Selected physical device {physical_device:?} and its queue family at index {queue_family_index}.");

        let (device, queues_iter) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        trace!("Created device.");
        let queues = queues_iter.collect::<Vec<_>>().into_boxed_slice();

        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        Ok(Self {
            device,
            queue_family_index,
            queues,
            next_queue: Arc::new(Mutex::new(0)),
            memory_allocator,
            command_buffer_allocator,
            buffers: Vec::new(),
        })
    }
}

impl Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device)
            .field("queue_family_index", &self.queue_family_index)
            .field("queues", &self.queues)
            .field("next_queue", &self.next_queue)
            .field("memory_allocator", &self.memory_allocator)
            .field("command_buffer_allocator", &"_")
            .finish()
    }
}

#[derive(Default, PartialEq)]
pub enum DeviceSelector<F>
where
    // I hate this so much but I honestly don't know if collecting them into a vec and passing that
    // would be better.
    F: FnOnce(&mut dyn Iterator<Item = Arc<PhysicalDevice>>) -> (Arc<PhysicalDevice>, u32),
{
    #[default]
    HighPower,
    LowPower,
    Custom(F),
}

#[derive(Debug, Default)]
pub enum QueueFamilySelector {
    // This assumes that the implementation will put its preferred queue families first.
    #[default]
    First,
    MostQueues,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("device selector condition failed")]
    DeviceSelectorFailed,
    #[error("{index} is not an index for a valid queue family for physical device {device:?}")]
    InvalidQueueFamilyIndex {
        index: u32,
        device: Arc<PhysicalDevice>,
    },
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

fn select_device_and_queue_family(
    physical_devices: impl Iterator<Item = Arc<PhysicalDevice>>,
    device_selector: DeviceSelector<
        impl FnOnce(&mut dyn Iterator<Item = Arc<PhysicalDevice>>) -> (Arc<PhysicalDevice>, u32),
    >,
    queue_family_selector: QueueFamilySelector,
) -> Result<(Arc<PhysicalDevice>, u32), Error> {
    match device_selector {
        DeviceSelector::Custom(f) => {
            let mut capture = physical_devices;
            let res = f(&mut capture);
            if res.1 >= res.0.queue_family_properties().len() as u32 {
                Err(Error::InvalidQueueFamilyIndex {
                    index: res.1,
                    device: res.0,
                })
            } else {
                Ok(res)
            }
        }
        // This is probably a crap way of doing this but that can come back later.
        _ => {
            let device_qfi_iter = physical_devices.filter_map(|pd| {
                let mut queue_family_iter = pd.queue_family_properties()
                    .iter()
                    .enumerate()
                    .filter(|(i, qfp)| {
                        let res = qfp.queue_flags.contains(QueueFlags::COMPUTE);
                        if res {
                            trace!("Found computing queue family ({qfp:?} at idx {i}) for device {pd:?}.")
                        } else {
                            trace!("Found queue family ({qfp:?} at idx {i}) for device {pd:?} but it did not
                                support computing workloads.");
                        }
                        res
                    });
                let res = match &queue_family_selector {
                    QueueFamilySelector::First => queue_family_iter.next(),
                    QueueFamilySelector::MostQueues => {
                        queue_family_iter.max_by_key(|(_, qfp)| qfp.queue_count)
                    }
                }
                .map(|(qfi, _)| (pd.clone(), qfi as u32));
                if let Some((pd, qfi)) = res.as_ref() {
                    debug!("Found compute-capable device ({pd:?}) with nominated QFI {qfi}");
                } else {
                    debug!("Found device ({pd:?}) but it did not have any queue families that supported \
                        compute workloads.");
                }
                res
            });

            match device_selector {
                DeviceSelector::HighPower => {
                    device_qfi_iter.min_by_key(|(pd, _)| match pd.properties().device_type {
                        PhysicalDeviceType::DiscreteGpu => 0,
                        PhysicalDeviceType::IntegratedGpu => 1,
                        PhysicalDeviceType::VirtualGpu => 2,
                        PhysicalDeviceType::Cpu => 3,
                        PhysicalDeviceType::Other => 4,
                        _ => 5,
                    })
                }
                DeviceSelector::LowPower => {
                    device_qfi_iter.min_by_key(|(pd, _)| match pd.properties().device_type {
                        // TODO: These are somewhat arbitrary and should likely be revised.
                        // The idea currently is to prioritize power saving while still computing efficiently
                        // via gpu compute. Idk, might remove.
                        PhysicalDeviceType::IntegratedGpu => 0,
                        PhysicalDeviceType::VirtualGpu => 1,
                        PhysicalDeviceType::DiscreteGpu => 2,
                        PhysicalDeviceType::Cpu => 3,
                        PhysicalDeviceType::Other => 4,
                        _ => 5,
                    })
                }
                DeviceSelector::Custom(_) => unreachable!(),
            }
            .ok_or(Error::NoVulkanComputingDevices)
        }
    }
}
