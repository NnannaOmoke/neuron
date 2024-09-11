use std::sync::OnceLock;

pub const SHADER_MAIN_NAME: &'static str = "main";

#[derive(Default)]
pub struct Shaders {
    matmul: OnceLock<wgpu::ShaderModule>,
}

impl Shaders {
    pub fn check_matmul(&self) -> Option<&wgpu::ShaderModule> {
        self.matmul.get()
    }

    pub fn get_matmul(&self, device: &wgpu::Device) -> &wgpu::ShaderModule {
        self.matmul.get_or_init(|| init_matmul(device))
    }

    pub fn new() -> Self {
        Self::default()
    }
}

pub fn init_matmul(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::include_wgsl!("shaders/matmul.wgsl"))
}
