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
        Device, DeviceCreateInfo, DeviceOwned, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    library::VulkanLibrary,
    memory::allocator::StandardMemoryAllocator,
};

#[derive(Clone)]
pub struct GpuContext {
    device: Arc<Device>,
    queue_family_index: u32,
    queues: Box<[Arc<Queue>]>,
    next_queue: Arc<Mutex<usize>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    // This is problematic as a lot of stuff uses SubBuffer's instead which are genericly
    // typed which itself is a problem here. We'll see if this is really necessary.
    // held_buffers: Vec<Arc<Buffer>>,
}

impl GpuContext {
    pub fn from_raw_parts(parts: GpuContextParts) -> Result<Self, Error> {
        fn e(
            device: Arc<Device>,
            got: &Arc<Device>,
            arg: &'static str,
        ) -> Result<GpuContext, Error> {
            Err(Error::WrongDevice {
                expected_device: device,
                got_device: got.clone(),
                offending_argument: Some(arg),
            })
        }

        if !parts
            .device
            .active_queue_family_indices()
            .contains(&parts.queue_family_index)
        {
            return Err(Error::InvalidQueueFamilyIndexDevice {
                index: parts.queue_family_index,
                device: parts.device,
            });
        }
        for queue in parts.queues.iter().cloned() {
            if queue.device() != &parts.device {
                return e(parts.device, queue.device(), "queues");
            }
        }
        if parts.memory_allocator.device() != &parts.device {
            return e(
                parts.device,
                parts.memory_allocator.device(),
                "memory_allocator",
            );
        }
        if parts.command_buffer_allocator.device() != &parts.device {
            return e(
                parts.device,
                parts.command_buffer_allocator.device(),
                "command_buffer_allocator",
            );
        }

        Ok(Self {
            device: parts.device,
            queue_family_index: parts.queue_family_index,
            queues: parts.queues,
            next_queue: parts.next_queue,
            memory_allocator: parts.memory_allocator,
            command_buffer_allocator: parts.command_buffer_allocator,
        })
    }

    pub fn into_raw_parts(self) -> GpuContextParts {
        GpuContextParts {
            device: self.device,
            queue_family_index: self.queue_family_index,
            queues: self.queues,
            next_queue: self.next_queue,
            memory_allocator: self.memory_allocator,
            command_buffer_allocator: self.command_buffer_allocator,
        }
    }

    pub fn new(
        device_preference: DeviceSelector<
            impl FnOnce(&mut dyn Iterator<Item = Arc<PhysicalDevice>>) -> (Arc<PhysicalDevice>, u32),
        >,
        queue_family_preference: QueueFamilySelector,
    ) -> Result<Self, Error> {
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

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        Ok(Self {
            device,
            queue_family_index,
            queues,
            next_queue: Arc::new(Mutex::new(0)),
            memory_allocator,
            command_buffer_allocator,
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

#[derive(Clone)]
pub struct GpuContextParts {
    pub device: Arc<Device>,
    pub queue_family_index: u32,
    pub queues: Box<[Arc<Queue>]>,
    pub next_queue: Arc<Mutex<usize>>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl Debug for GpuContextParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContextParts")
            .field("device", &self.device)
            .field("queue_family_index", &self.queue_family_index)
            .field("queues", &self.queues)
            .field("next_queue", &self.next_queue)
            .field("memory_allocator", &self.memory_allocator)
            .field("command_buffer_allocator", &"_")
            .finish()
    }
}

#[derive(Clone, Default, PartialEq)]
pub enum DeviceSelector<F>
where
    // I hate this so much but I honestly don't know if collecting them into a vec and
    // passing that would be better.
    F: FnOnce(&mut dyn Iterator<Item = Arc<PhysicalDevice>>) -> (Arc<PhysicalDevice>, u32),
{
    #[default]
    HighPower,
    LowPower,
    Custom(F),
}

impl<F> Debug for DeviceSelector<F>
where
    F: FnOnce(&mut dyn Iterator<Item = Arc<PhysicalDevice>>) -> (Arc<PhysicalDevice>, u32),
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighPower => f.write_str("HighPower"),
            Self::LowPower => f.write_str("LowPower"),
            Self::Custom(_) => f.write_str("Custom({<F>}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
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
    #[error(
        "{index} is not an index for a valid queue family for physical device {physical_device:?}"
    )]
    InvalidQueueFamilyIndexPhysicalDevice {
        index: u32,
        physical_device: Arc<PhysicalDevice>,
    },
    #[error("{index} is not an index for a valid queue family for vulkano device {device:?}")]
    InvalidQueueFamilyIndexDevice { index: u32, device: Arc<Device> },
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
    #[error("device of provided value was not the expected device")]
    WrongDevice {
        expected_device: Arc<Device>,
        got_device: Arc<Device>,
        /// (If applicable) the argument, field, etc whose passed value
        offending_argument: Option<&'static str>,
    },
}

pub fn default_custom_target(
    physical_devices: &mut dyn Iterator<Item = Arc<PhysicalDevice>>,
) -> (Arc<PhysicalDevice>, u32) {
    // Yanked from select_device_and_queue_family.
    physical_devices
        .filter_map(|pd| {
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
            let res = queue_family_iter
                .next()
                .map(|(qfi, _)| (pd.clone(), qfi as u32));
            if let Some((pd, qfi)) = res.as_ref() {
                debug!("Found compute-capable device ({pd:?}) with nominated QFI {qfi}");
            } else {
                debug!(
                    "Found device ({pd:?}) but it did not have any queue families that supported \
                compute workloads."
                );
            }
            res
        })
        // TODO: Consider changing desired closure type to return Option<_> instead. I don't like this unwrap.
        .next()
        .unwrap()
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
                Err(Error::InvalidQueueFamilyIndexPhysicalDevice {
                    index: res.1,
                    physical_device: res.0,
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

pub type DeviceSelectorCustomFnPtr =
    fn(&mut dyn Iterator<Item = Arc<PhysicalDevice>>) -> (Arc<PhysicalDevice>, u32);

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use vulkano::{
        buffer::{BufferCreateFlags, BufferCreateInfo, BufferUsage},
        memory::allocator::{AllocationCreateInfo, DeviceLayout},
    };

    use super::*;

    #[test]
    fn gpu_context_new() {
        for device_selector in [
            DeviceSelector::HighPower,
            DeviceSelector::LowPower,
            DeviceSelector::Custom(|i| (i.next().unwrap(), 0)),
        ] {
            for queue_family_selector in
                [QueueFamilySelector::First, QueueFamilySelector::MostQueues]
            {
                println!("Attempting to create GpuContext using device selector {:?} and queue family selector \
                    {queue_family_selector:?}...", device_selector.clone());
                GpuContext::new(device_selector.clone(), queue_family_selector).unwrap();
            }
        }
    }

    #[test]
    fn gpu_context_from_raw_parts() {
        // Assumes into_raw_parts does what it's supposed to bc it's that simple of a fn.
        #[inline]
        fn c() -> GpuContextParts {
            GpuContext::new(
                DeviceSelector::<DeviceSelectorCustomFnPtr>::HighPower,
                QueueFamilySelector::First,
            )
            .unwrap()
            .into_raw_parts()
        }

        #[inline]
        fn was_ok() {
            panic!("should have paniced but didn't");
        }

        #[inline]
        fn wrong_error_type(e: Error) {
            panic!("from_raw_parts returned the wrong variant of error ({e:?})");
        }

        #[inline]
        fn expect_wrong_device(parts: GpuContextParts, arg: &'static str) {
            match GpuContext::from_raw_parts(parts) {
                Ok(_) => was_ok(),
                Err(e) => {
                    if let Error::WrongDevice {
                        offending_argument, ..
                    } = e
                    {
                        assert_eq!(offending_argument.unwrap(), arg);
                    } else {
                        wrong_error_type(e);
                    }
                }
            }
        }

        #[inline]
        fn expect_bad_qfi(parts: GpuContextParts) {
            match GpuContext::from_raw_parts(parts) {
                Ok(_) => was_ok(),
                Err(e) => match e {
                    Error::InvalidQueueFamilyIndexDevice { .. } => {}
                    o => wrong_error_type(o),
                },
            }
        }

        let other_ctx = c();
        let basectx = c();
        let mut testctx = basectx.clone();

        testctx.queue_family_index = u32::MAX;
        expect_bad_qfi(testctx);

        testctx = basectx.clone();
        testctx.queues = other_ctx.queues;
        expect_wrong_device(testctx, "queues");

        testctx = basectx.clone();
        testctx.memory_allocator = other_ctx.memory_allocator;
        expect_wrong_device(testctx, "memory_allocator");

        testctx = basectx.clone();
        testctx.command_buffer_allocator = other_ctx.command_buffer_allocator;
        expect_wrong_device(testctx, "command_buffer_allocator");
    }
}
