/// TODO: Add docs.
use super::{context::GpuContext, Error};

use std::{ops::Deref, sync::Arc};
use vulkano::{buffer::Buffer, device::DeviceOwned};

pub struct ComputeContext<GpuContextPtr: Deref<Target = GpuContext>> {
    gpu_context: GpuContextPtr,
    input_buffers: Vec<Arc<Buffer>>,
    output_buffers: Vec<Arc<Buffer>>,
    support_buffers: Option<Vec<Arc<Buffer>>>,
}

impl<GpuContextPtr: Deref<Target = GpuContext>> ComputeContext<GpuContextPtr> {
    pub fn from_raw_parts(parts: ComputeContextParts<GpuContextPtr>) -> Result<Self, Error> {
        for buffer in parts.input_buffers.iter() {
            if buffer.device() != parts.gpu_context.device() {
                return Err(Error::WrongDevice {
                    expected_device: parts.gpu_context.device().clone(),
                    got_device: buffer.device().clone(),
                    // TODO: Make the offending_argument thing take a string so we can do support_buffers[i]
                    offending_argument: Some("output_buffers"),
                });
            }
        }
        for buffer in parts.output_buffers.iter() {
            if buffer.device() != parts.gpu_context.device() {
                return Err(Error::WrongDevice {
                    expected_device: parts.gpu_context.device().clone(),
                    got_device: buffer.device().clone(),
                    // TODO: Make the offending_argument thing take a string so we can do support_buffers[i]
                    offending_argument: Some("output_buffers"),
                });
            }
        }
        for buffer in parts.support_buffers.iter().flatten() {
            if buffer.device() != parts.gpu_context.device() {
                return Err(Error::WrongDevice {
                    expected_device: parts.gpu_context.device().clone(),
                    got_device: buffer.device().clone(),
                    // TODO: Make the offending_argument thing take a string so we can do support_buffers[i]
                    offending_argument: Some("support_buffers"),
                });
            }
        }

        Ok(Self::from_raw_parts_unchecked(parts))
    }

    pub fn from_raw_parts_unchecked(
        ComputeContextParts {
            gpu_context,
            input_buffers,
            output_buffers,
            support_buffers,
        }: ComputeContextParts<GpuContextPtr>,
    ) -> Self {
        ComputeContext {
            gpu_context,
            input_buffers,
            output_buffers,
            support_buffers,
        }
    }
}

pub struct ComputeContextParts<GpuContextPtr: Deref<Target = GpuContext>> {
    pub gpu_context: GpuContextPtr,
    pub input_buffers: Vec<Arc<Buffer>>,
    pub output_buffers: Vec<Arc<Buffer>>,
    pub support_buffers: Option<Vec<Arc<Buffer>>>,
}
