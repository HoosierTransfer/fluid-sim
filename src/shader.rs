use gl::types::*;
use std::fs;

pub enum ShaderType {
    Vertex,
    Fragment,
}

pub struct Shader {
    pub id: u32,

    vertex_code: String,
    fragment_code: String,

    vertex_defines: Vec<(String, String)>,
    fragment_defines: Vec<(String, String)>,

    compiled: bool,
}

impl Shader {
    pub fn new(shader_name: &str) -> Self {
        let vertex_path = format!("assets/shaders/{}.vsh", shader_name);
        let fragment_path = format!("assets/shaders/{}.fsh", shader_name);

        let vertex_code = fs::read_to_string(vertex_path).expect("Failed to read vertex shader");
        let fragment_code = fs::read_to_string(fragment_path).expect("Failed to read fragment shader");

        let vertex_defines = Self::extract_defines(&vertex_code);
        let fragment_defines = Self::extract_defines(&fragment_code);

        Shader {
            id: 0,
            vertex_code,
            fragment_code,
            vertex_defines,
            fragment_defines,
            compiled: false,
        }
    }

    pub fn compile(&mut self) {
        let c_str_vert: std::ffi::CString = std::ffi::CString::new(self.vertex_code.as_bytes()).unwrap();
        let c_str_frag: std::ffi::CString = std::ffi::CString::new(self.fragment_code.as_bytes()).unwrap();

        let vertex_shader;
        let fragment_shader;
        let shader_program;

        unsafe {
            vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            gl::ShaderSource(vertex_shader, 1, &c_str_vert.as_ptr(), std::ptr::null());
            gl::CompileShader(vertex_shader);

            let mut success = gl::FALSE as GLint;
            gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                let mut len = 0;
                gl::GetShaderiv(vertex_shader, gl::INFO_LOG_LENGTH, &mut len);
                let mut buf = vec![0; len as usize];
                gl::GetShaderInfoLog(vertex_shader, len, std::ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
                panic!(
                    "{}",
                    std::str::from_utf8(&buf)
                        .map_err(|_| "ShaderInfoLog not valid utf8")
                        .unwrap()
                );
            }

            fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(fragment_shader, 1, &c_str_frag.as_ptr(), std::ptr::null());
            gl::CompileShader(fragment_shader);

            let mut success = gl::FALSE as GLint;
            gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                let mut len = 0;
                gl::GetShaderiv(fragment_shader, gl::INFO_LOG_LENGTH, &mut len);
                let mut buf = vec![0; len as usize];
                gl::GetShaderInfoLog(fragment_shader, len, std::ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
                panic!(
                    "{}",
                    std::str::from_utf8(&buf)
                        .map_err(|_| "ShaderInfoLog not valid utf8")
                        .unwrap()
                );
            }

            shader_program = gl::CreateProgram();
            gl::AttachShader(shader_program, vertex_shader);
            gl::AttachShader(shader_program, fragment_shader);
            gl::LinkProgram(shader_program);

            let mut success = gl::FALSE as GLint;
            gl::GetProgramiv(shader_program, gl::LINK_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                let mut len = 0;
                gl::GetProgramiv(shader_program, gl::INFO_LOG_LENGTH, &mut len);
                let mut buf = vec![0; len as usize];
                gl::GetProgramInfoLog(shader_program, len, std::ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
                panic!(
                    "{}",
                    std::str::from_utf8(&buf)
                        .map_err(|_| "ShaderInfoLog not valid utf8")
                        .unwrap()
                );
            }
        }

        self.id = shader_program;

        unsafe {
            gl::DeleteShader(vertex_shader);
            gl::DeleteShader(fragment_shader);
        }
    }

    pub fn add_define(&mut self, name: &str, value: &str, shader_type: ShaderType) {
        let add_or_replace_define = |mut shader_code: String| {
            if let Some(pos) = shader_code.find("#version") {
                if let Some(new_line_pos) = shader_code[pos..].find('\n') {
                    let insert_pos = pos + new_line_pos + 1;
                    let define_string = format!("#define {} {}", name, value);
                    if let Some(define_pos) = shader_code.find(&define_string) {
                        let end = shader_code[define_pos..].find('\n').unwrap_or(shader_code.len());
                        shader_code.replace_range(define_pos..define_pos + end, &define_string);
                    } else {
                        shader_code.insert_str(insert_pos, &format!("{}\n", define_string));
                    }
                }
            }
        };

        match shader_type {
            ShaderType::Vertex => {
                add_or_replace_define(self.vertex_code.clone());
                self.vertex_defines.push((name.to_string(), value.to_string()));
            },
            ShaderType::Fragment => {
                add_or_replace_define(self.fragment_code.clone());
                self.fragment_defines.push((name.to_string(), value.to_string()));
            },
        }
    }

    pub fn remove_define(&mut self, name: &str, shader_type: ShaderType) {
        let remove_define = |mut shader_code: String| {
            if let Some(pos) = shader_code.find(format!("#define {}", name).as_str()) {
                let end = shader_code[pos..].find('\n').unwrap_or(shader_code.len());
                shader_code.replace_range(pos..pos + end, "");
            }
        };

        match shader_type {
            ShaderType::Vertex => {
                remove_define(self.vertex_code.clone());
                self.vertex_defines.retain(|(n, _)| n != name);
            },
            ShaderType::Fragment => {
                remove_define(self.fragment_code.clone());
                self.fragment_defines.retain(|(n, _)| n != name);
            },
        }
    }

    pub fn get_define(&self, name: &str, shader_type: ShaderType) -> Option<&str> {
        match shader_type {
            ShaderType::Vertex => {
                self.vertex_defines.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
            },
            ShaderType::Fragment => {
                self.fragment_defines.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
            },
        }
    }

    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    pub fn use_program(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }

    pub fn set_bool(&self, name: &str, value: bool) {
        unsafe {
            gl::Uniform1i(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), value as i32);
        }
    }

    pub fn set_int(&self, name: &str, value: i32) {
        unsafe {
            gl::Uniform1i(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), value);
        }
    }

    pub fn set_float(&self, name: &str, value: f32) {
        unsafe {
            gl::Uniform1f(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), value);
        }
    }

    pub fn set_vec2(&self, name: &str, value: [f32; 2]) {
        unsafe {
            gl::Uniform2fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), 1, value.as_ptr());
        }
    }

    pub fn set_vec3(&self, name: &str, value: [f32; 3]) {
        unsafe {
            gl::Uniform3fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), 1, value.as_ptr());
        }
    }

    pub fn set_vec4(&self, name: &str, value: [f32; 4]) {
        unsafe {
            gl::Uniform4fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), 1, value.as_ptr());
        }
    }

    pub fn set_mat2(&self, name: &str, value: nalgebra::Matrix2<f32>) {
        unsafe {
            gl::UniformMatrix2fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), 1, gl::FALSE, value.as_slice().as_ptr());
        }
    }

    pub fn set_mat3(&self, name: &str, value: nalgebra::Matrix3<f32>) {
        unsafe {
            gl::UniformMatrix3fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), 1, gl::FALSE, value.as_slice().as_ptr());
        }
    }

    pub fn set_mat4(&self, name: &str, value: nalgebra::Matrix4<f32>) {
        unsafe {
            gl::UniformMatrix4fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(name).unwrap().as_ptr()), 1, gl::FALSE, value.as_slice().as_ptr());
        }
    }

    pub fn set_bool_in_array(&self, name: &str, value: bool, index: usize) {
        unsafe {
            gl::Uniform1i(gl::GetUniformLocation(self.id, std::ffi::CString::new(format!("{}[{}]", name, index)).unwrap().as_ptr()), value as i32);
        }
    }

    pub fn set_int_in_array(&self, name: &str, value: i32, index: usize) {
        unsafe {
            gl::Uniform1i(gl::GetUniformLocation(self.id, std::ffi::CString::new(format!("{}[{}]", name, index)).unwrap().as_ptr()), value);
        }
    }

    pub fn set_float_in_array(&self, name: &str, value: f32, index: usize) {
        unsafe {
            gl::Uniform1f(gl::GetUniformLocation(self.id, std::ffi::CString::new(format!("{}[{}]", name, index)).unwrap().as_ptr()), value);
        }
    }

    pub fn set_vec2_in_array(&self, name: &str, value: [f32; 2], index: usize) {
        unsafe {
            gl::Uniform2fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(format!("{}[{}]", name, index)).unwrap().as_ptr()), 1, value.as_ptr());
        }
    }

    pub fn set_vec3_in_array(&self, name: &str, value: [f32; 3], index: usize) {
        unsafe {
            gl::Uniform3fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(format!("{}[{}]", name, index)).unwrap().as_ptr()), 1, value.as_ptr());
        }
    }

    pub fn set_vec4_in_array(&self, name: &str, value: [f32; 4], index: usize) {
        unsafe {
            gl::Uniform4fv(gl::GetUniformLocation(self.id, std::ffi::CString::new(format!("{}[{}]", name, index)).unwrap().as_ptr()), 1, value.as_ptr());
        }
    }

    fn extract_defines(shader_code: &str) -> Vec<(String, String)> {
        let mut defines = Vec::new();
        let mut pos = 0;
    
        while let Some(start) = shader_code[pos..].find("#define ") {
            pos += start;
            let end = shader_code[pos..].find('\n').unwrap_or(shader_code.len() - pos) + pos;
            let define = &shader_code[pos..end];
    
            let mut parts = define.split_whitespace();
            let _ = parts.next();
            let name = parts.next().unwrap_or("").to_string();
            let value = parts.collect::<Vec<&str>>().join(" ");
    
            defines.push((name, value));
            pos = end + 1;
        }
    
        defines
    }
}