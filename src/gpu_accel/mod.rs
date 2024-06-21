pub mod compute;
pub mod context;

use std::{fmt::Debug, sync::Arc};
// use thiserror::Error;

// #[derive(Debug, Error)]
// pub enum Error {
//     #[error("device selector condition failed")]
//     DeviceSelectorFailed,
//     #[error(
//         "{index} is not an index for a valid queue family for physical device {physical_device:?}"
//     )]
//     InvalidQueueFamilyIndexPhysicalDevice {
//         index: u32,
//         physical_device: Arc<PhysicalDevice>,
//     },
//     #[error("{index} is not an index for a valid queue family for vulkano device {device:?}")]
//     InvalidQueueFamilyIndexDevice { index: u32, device: Arc<Device> },
//     #[error(transparent)]
//     LibraryLoading(#[from] vulkano::library::LoadingError),
//     #[error("no physical devices that support Vulkan also support computing")]
//     NoVulkanComputingDevices,
//     #[error("no physical devices that support Vulkan were found")]
//     NoVulkanDevices,
//     #[error(transparent)]
//     VulkanValidated(#[from] vulkano::Validated<vulkano::VulkanError>),
//     #[error(transparent)]
//     Vulkan(#[from] vulkano::VulkanError),
//     #[error("device of provided value was not the expected device")]
//     WrongDevice {
//         expected_device: Arc<Device>,
//         got_device: Arc<Device>,
//         /// (If applicable) the argument, field, etc whose passed value
//         offending_argument: Option<&'static str>,
//     },
// }
