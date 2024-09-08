use super::context::GpuContext;
use ndarray::{ArrayView2, ArrayViewMut2};
use std::{borrow::Borrow, ops::Deref};

pub struct MatMulContext<GpuCtxPtr: Borrow<GpuContext>> {
    context_ptr: GpuCtxPtr,
    lhs_dims: (usize, usize),
    lhs_buf: wgpu::Buffer,
    rhs_dims: (usize, usize),
    rhs_buf: wgpu::Buffer,
    output_staging_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
}

impl<GpuCtxSrc: Borrow<GpuContext>> MatMulContext<GpuCtxSrc> {
    pub fn new(gpu_context_ptr: GpuCtxSrc, ) -> Self {
        let buffer_descriptor = wgpu::BufferDescriptor {
            label: Some("neuron DotContext out buffer"),
            size: (matrix_size_in_f32s * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let lhs_buf = gpu_context_ptr
            .borrow()
            .device()
            .create_buffer(&buffer_descriptor);
        let rhs_buf = gpu_context_ptr
            .borrow()
            .device()
            .create_buffer(&buffer_descriptor);
        let output_staging_buf = gpu_context_ptr
            .borrow()
            .device()
            .create_buffer(&buffer_descriptor);

        Self {
            context_ptr: gpu_context_ptr,
            matrix_size: matrix_size_in_f32s,
            lhs_buf,
            rhs_buf,
            output_staging_buf,
            out_buf: None,
        }
    }

    /// TODO
    ///
    /// ## Panics
    ///
    /// Panics if the dimentions of `lhs_and_output` and `rhs` are not the same.
    /// Also panics if the number of elements held in both of the matrices is not the
    /// same as the number of elements the `DotContext` was created to work with.
    ///
    /// ## TODO
    ///
    /// Make this method work with matrices smaller than the max size held by the
    /// context when the `DotContext`'s size is a maximum rather than an exact value.
    pub async fn dot_in_place(
        &self,
        mut lhs_and_output: ArrayViewMut2<'_, f32>,
        rhs: ArrayView2<'_, f32>,
    ) -> Result<(), Error> {
        assert_eq!(lhs_and_output.dim(), rhs.dim());
        assert_eq!(rhs.len(), self.matrix_size);

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {}
