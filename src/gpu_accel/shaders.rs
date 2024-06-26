use std::sync::OnceLock;

#[derive(Default)]
pub struct Shaders {
    dot_in_place: OnceLock<wgpu::ShaderModule>,
    dot_extern: OnceLock<wgpu::ShaderModule>,
}

impl Shaders {
    pub fn check_dot_in_place(&self) -> Option<&wgpu::ShaderModule> {
        self.dot_in_place.get()
    }

    pub fn check_dot_extern(&self) -> Option<&wgpu::ShaderModule> {
        self.dot_extern.get()
    }

    pub fn get_dot_in_place(&self, device: &wgpu::Device) -> &wgpu::ShaderModule {
        self.dot_in_place.get_or_init(|| init_dot_in_place(device))
    }

    pub fn get_dot_extern(&self,device: &wgpu::Device) -> &wgpu::ShaderModule {
        self.dot_in_place.get_or_init(|| init_dot_extern(device))
    }
    
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn init_dot_in_place(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::include_wgsl!("shaders/dot_in_place.wgsl"))
}

pub fn init_dot_extern(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::include_wgsl!("shaders/dot_extern.wgsl"))
}
