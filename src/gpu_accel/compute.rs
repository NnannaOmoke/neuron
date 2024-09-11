use super::context::GpuContext;
use ndarray::{ArrayView2, ArrayViewMut2};
use std::{borrow::Borrow, ops::Deref};

pub struct MatMulContext<GpuCtxPtr: Borrow<GpuContext>> {
    context_ptr: GpuCtxPtr,
    lhs_dim: (usize, usize),
    lhs_buf: wgpu::Buffer,
    lhs_staging_buf: wgpu::Buffer,
    rhs_dim: (usize, usize),
    rhs_buf: wgpu::Buffer,
    // May not be needed if rhs is the same dim as lhs.
    rhs_staging_buf: Option<wgpu::Buffer>,
    out_buf: wgpu::Buffer,
    out_staging_buf: wgpu::Buffer,
}

impl<GpuCtxSrc: Borrow<GpuContext>> MatMulContext<GpuCtxSrc> {
    pub fn new(
        gpu_context_ptr: GpuCtxSrc,
        lhs_dim: (usize, usize),
        rhs_dim: (usize, usize),
    ) -> Self {
        let device = gpu_context_ptr.borrow().device();

        let lhs_buf_size = (lhs_dim.0 * lhs_dim.1) as wgpu::BufferAddress;
        let lhs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neuron MatMulContext lhs buffer"),
            size: lhs_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lhs_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neuron MatMulContext lhs staging buffer"),
            size: lhs_buf_size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rhs_buf_size = (rhs_dim.0 * rhs_dim.1) as wgpu::BufferAddress;
        let rhs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neuron MatMulContext rhs buffer"),
            size: rhs_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rhs_staging_buf = if lhs_dim.0 == rhs_dim.0 && lhs_dim.1 == rhs_dim.1 {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("neuron MatMulContext rhs staging buffer"),
                size: rhs_buf_size,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let out_dim = (lhs_dim.0, rhs_dim.1);
        let out_buf_size = (out_dim.0 * out_dim.1) as wgpu::BufferAddress;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neuron MatMulContext output buffer"),
            size: out_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neuron MatMulContext output staging buffer"),
            size: out_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            context_ptr: gpu_context_ptr,
            lhs_dim,
            lhs_buf,
            lhs_staging_buf,
            rhs_dim,
            rhs_buf,
            rhs_staging_buf,
            out_buf,
            out_staging_buf,
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
