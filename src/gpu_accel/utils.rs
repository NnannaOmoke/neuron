use ndarray::{ArrayView2, ArrayViewMut2};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("matrices were not of compatible dimentions")]
    BadMatrixSizes,
    #[error(transparent)]
    BufferAsyncError(#[from] wgpu::BufferAsyncError),
    #[error(transparent)]
    TokioOneshotChannelReveiveError(#[from] tokio::sync::oneshot::error::RecvError),
}

pub fn get_needed_buffer_size(item_count: usize) -> u64 {
    (item_count * size_of::<f32>()) as u64
}

fn create_loadable_buffer(
    device: &wgpu::Device,
    data_len: usize,
    needs_cpy_src: bool,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (data_len * size_of::<f32>()) as wgpu::BufferAddress,
        usage: if needs_cpy_src {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE
        },
        mapped_at_creation: true,
    })
}

pub async fn get_mapped_range_mut<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
) -> Result<wgpu::BufferViewMut<'a>, Error> {
    let buffer_slice = buffer.slice(..);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Write, move |r| {
        sender
            .send(r)
            .expect("failed to send buffer mapping result through channel")
    });
    device.poll(wgpu::Maintain::Poll);
    receiver.await??;
    Ok(buffer_slice.get_mapped_range_mut())
}

/// Alternative for wgpu::util::DeviceExt::create_buffer_init specialized for ndarray array views.
///
/// The issue with wgpu's `create_buffer_init` is that it only works for data from a slice.
/// The issue with that is that ndarray `ArrayView`s can't always be converted to slices due to
/// internal memory representation. This function allows us to effectively do the same as the
/// provided convenince method but fall back to iterative copying if the `ArrayView` cannot be
/// reborrowed as a slice.
pub async fn create_loaded_buffer(
    device: &wgpu::Device,
    data: &ArrayView2<'_, f32>,
    needs_cpy_src: bool,
    data_name: &str,
) -> Result<wgpu::Buffer, Error> {
    let buffer = create_loadable_buffer(device, data.len(), needs_cpy_src);

    let mut buffer_view_mut = get_mapped_range_mut(device, &buffer).await?;
    if let Some(data_slice) = data.as_slice() {
        buffer_view_mut
            .copy_from_slice(bytemuck::cast_slice(data_slice));
    } else {
        log::warn!(
            "{} cannot be cast to slice (ndarray::ArrayView::to_slice). \
            Falling back in per-float insertion. This is significantly slower.",
            data_name
        );
        for (i, e) in data.iter().enumerate() {
            buffer_view_mut[i..i + size_of::<f32>()].copy_from_slice(&e.to_le_bytes());
        }
    }

    drop(buffer_view_mut);
    buffer.unmap();
    Ok(buffer)
}

/// See docs for [`create_loaded_buffer`].
pub async fn create_loaded_buffer_from_mut(
    device: &wgpu::Device,
    data: &ArrayViewMut2<'_, f32>,
    needs_cpy_src: bool,
    data_name: &str,
) -> Result<wgpu::Buffer, Error> {
    let buffer = create_loadable_buffer(device, data.len(), needs_cpy_src);

    let mut buffer_view_mut = get_mapped_range_mut(device, &buffer).await?;
    if let Some(data_slice) = data.as_slice() {
        buffer_view_mut
            .copy_from_slice(bytemuck::cast_slice(data_slice));
    } else {
        log::warn!(
            "{} cannot be cast to slice (ndarray::ArrayView::to_slice). \
            Falling back in per-float insertion. This is significantly slower.",
            data_name
        );
        for (i, e) in data.iter().enumerate() {
            buffer_view_mut[i..i + size_of::<f32>()].copy_from_slice(&e.to_le_bytes());
        }
    }

    drop(buffer_view_mut);
    buffer.unmap();
    Ok(buffer)
}

