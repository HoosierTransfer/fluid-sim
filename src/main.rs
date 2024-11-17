use glutin::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
use glutin::event_loop::{ControlFlow, EventLoop};
use glutin::window::WindowBuilder;
use glutin::ContextBuilder;
use nalgebra::Vector3;

use std::time::Instant;
use std::io::Read;
use std::collections::HashSet;
use std::ffi::c_void;
use std::ptr;

use rand::Rng;

use glutin::dpi::PhysicalPosition;


mod shader;

fn diffuse(b: i32, x: &mut Vec<f32>, x0: &Vec<f32>, diff: f32, dt: f32, iter: i32, n: usize) {
    let a = dt * diff * (n as f32 - 2.0) * (n as f32 - 2.0);
    lin_solve(b, x, x0, a, 1.0 + 6.0 * a, iter, n);
}

fn lin_solve(b: i32, x: &mut Vec<f32>, x0: &Vec<f32>, a: f32, c: f32, iter: i32, n: usize) {
    let c_recip = 1.0 / c;
    for _ in 0..iter {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let idx = i + j * n;
                x[idx] = (x0[idx]
                    + a * (x[idx - 1] + x[idx + 1] + x[idx - n] + x[idx + n]))
                    * c_recip;
            }
        }
        set_bnd(b, x, n);
    }
}

fn project(vx: &mut Vec<f32>, vy: &mut Vec<f32>, p: &mut Vec<f32>, div: &mut Vec<f32>, n: usize) {
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let idx = i + j * n;
            div[idx] = -0.5 * (vx[idx + 1] - vx[idx - 1] + vy[idx + n] - vy[idx - n]) / n as f32;
            p[idx] = 0.0;
        }
    }

    set_bnd(0, div, n);
    set_bnd(0, p, n);
    lin_solve(0, p, div, 1.0, 6.0, 4, n);

    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let idx = i + j * n;
            vx[idx] -= 0.5 * (p[idx + 1] - p[idx - 1]) * n as f32;
            vy[idx] -= 0.5 * (p[idx + n] - p[idx - n]) * n as f32;
        }
    }

    set_bnd(1, vx, n);
    set_bnd(2, vy, n);
}

fn advect(
    b: i32,
    d: &mut Vec<f32>,
    d0: &Vec<f32>,
    vx: &Vec<f32>,
    vy: &Vec<f32>,
    dt: f32,
    n: usize,
) {
    let dtx = dt * (n as f32 - 2.0);
    let dty = dt * (n as f32 - 2.0);

    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let idx = i + j * n;

            let mut x = i as f32 - dtx * vx[idx];
            let mut y = j as f32 - dty * vy[idx];

            if x < 0.5 {
                x = 0.5;
            }
            if x > n as f32 - 1.5 {
                x = n as f32 - 1.5;
            }
            let i0 = x.floor() as usize;
            let i1 = i0 + 1;

            if y < 0.5 {
                y = 0.5;
            }
            if y > n as f32 - 1.5 {
                y = n as f32 - 1.5;
            }
            let j0 = y.floor() as usize;
            let j1 = j0 + 1;

            let s1 = x - i0 as f32;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f32;
            let t0 = 1.0 - t1;

            d[idx] = s0 * (t0 * d0[i0 + j0 * n] + t1 * d0[i0 + j1 * n])
                + s1 * (t0 * d0[i1 + j0 * n] + t1 * d0[i1 + j1 * n]);
        }
    }
    set_bnd(b, d, n);
}

// Replace the set_bnd function with this corrected version
fn set_bnd(b: i32, x: &mut Vec<f32>, n: usize) {
    // For velocity components, reverse the direction at boundaries
    for i in 1..n - 1 {
        // Top and bottom walls
        x[i] = if b == 2 { -x[i + n] } else { x[i + n] }; // Top wall
        x[i + (n-1) * n] = if b == 2 { -x[i + (n-2) * n] } else { x[i + (n-2) * n] }; // Bottom wall
        
        // Left and right walls
        x[i * n] = if b == 1 { -x[1 + i * n] } else { x[1 + i * n] }; // Left wall
        x[(i+1) * n - 1] = if b == 1 { -x[(i+1) * n - 2] } else { x[(i+1) * n - 2] }; // Right wall
    }

    // Corner cases
    x[0] = 0.5 * (x[1] + x[n]); // Top-left
    x[n-1] = 0.5 * (x[n-2] + x[2*n-1]); // Top-right
    x[(n-1)*n] = 0.5 * (x[(n-2)*n] + x[(n-1)*n+1]); // Bottom-left
    x[n*n-1] = 0.5 * (x[n*n-2] + x[(n-1)*n-1]); // Bottom-right
}

struct FluidSolver {
    width: u32,
    height: u32,
    size: usize,
    density: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vx0: Vec<f32>,
    vy0: Vec<f32>,
    s: Vec<f32>,
    visc: f32,
    diff: f32,
    mouse_pos: (f32, f32),
    prev_mouse_pos: (f32, f32),
    mouse_down: bool,
    curl: Vec<f32>,
    vort_force: f32,
    right_mouse_down: bool,
}

impl FluidSolver {
    fn new(width: u32, height: u32) -> FluidSolver {
        let size = (width * height) as usize;

        FluidSolver {
            width,
            height,
            size,
            density: vec![0.0; size],
            vx: vec![0.0; size],
            vy: vec![0.0; size],
            vx0: vec![0.0; size],
            vy0: vec![0.0; size],
            s: vec![0.0; size],
            visc: 0.0,
            diff: 0.0,
            mouse_pos: (0.0, 0.0),
            prev_mouse_pos: (0.0, 0.0),
            mouse_down: false,
            curl: vec![0.0; size],
            vort_force: 0.1,
            right_mouse_down: false,
        }
    }

    fn compute_curl(&mut self) {
        let n = self.width as usize; // Use width instead of size
        
        for j in 1..self.height as usize - 1 {
            for i in 1..self.width as usize - 1 {
                let idx = i + j * n;
                
                // Calculate curl as dv/dx - du/dy
                let du_dy = (self.vx[idx + n] - self.vx[idx - n]) * 0.5;
                let dv_dx = (self.vy[idx + 1] - self.vy[idx - 1]) * 0.5;
                self.curl[idx] = dv_dx - du_dy;
            }
        }
    }

    fn apply_vorticity(&mut self, dt: f32) {
        let n = self.width as usize;
        
        for j in 1..self.height as usize - 1 {
            for i in 1..self.width as usize - 1 {
                let idx = i + j * n;
                
                let mut curl_grad_x = (self.curl[idx + 1].abs() - self.curl[idx - 1].abs()) * 0.5;
                let mut curl_grad_y = (self.curl[idx + n].abs() - self.curl[idx - n].abs()) * 0.5;
                
                // Normalize
                let length = (curl_grad_x * curl_grad_x + curl_grad_y * curl_grad_y).sqrt();
                if length > 0.0 {
                    curl_grad_x /= length;
                    curl_grad_y /= length;
                }
    
                // Apply force
                let force = self.vort_force * dt;
                self.vx[idx] += force * curl_grad_y * self.curl[idx];
                self.vy[idx] -= force * curl_grad_x * self.curl[idx];
            }
        }
    }

    fn add_density(&mut self, x: u32, y: u32, amount: f32) {
        let index = self.index(x, y);
        self.density[index] += amount;
    }

    fn add_velocity(&mut self, x: u32, y: u32, amount_x: f32, amount_y: f32) {
        let index = self.index(x, y);
        self.vx[index] += amount_x;
        self.vy[index] += amount_y;
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (x + y * self.width) as usize
    }

    fn step(&mut self, dt: f32) {
        // Pass width as n instead of size
        let n = self.width as usize;
        
        diffuse(1, &mut self.vx0, &self.vx, self.visc, dt, 4, n);
        diffuse(2, &mut self.vy0, &self.vy, self.visc, dt, 4, n);

        project(
            &mut self.vx0,
            &mut self.vy0, 
            &mut self.vx,
            &mut self.vy,
            n
        );

        advect(
            1,
            &mut self.vx,
            &self.vx0,
            &self.vx0,
            &self.vy0,
            dt,
            n
        );

        advect(
            2, 
            &mut self.vy,
            &self.vy0,
            &self.vx0,
            &self.vy0,
            dt,
            n
        );

        self.compute_curl();
        self.apply_vorticity(dt);

        project(
            &mut self.vx,
            &mut self.vy,
            &mut self.vx0,
            &mut self.vy0,
            n
        );

        diffuse(
            0,
            &mut self.s,
            &self.density,
            self.diff,
            dt,
            4,
            n
        );
        
        advect(
            0,
            &mut self.density,
            &self.s,
            &self.vx,
            &self.vy,
            dt,
            n
        );
        
        for i in 0..self.size {
            self.density[i] *= 0.995;
        }

        for i in 0..self.size {
            self.vx[i] *= 0.995;
            self.vy[i] *= 0.995;
        }
    }

    fn render(&self, data: &mut Vec<f32>) {
        for j in 0..self.height {
            for i in 0..self.width {
                let index = self.index(i, j);
                data[index] = self.density[index];
            }
        }
    }
}

fn main() {
    let el = EventLoop::new();
    let wb = WindowBuilder::new().with_title("WOW! so silly :3");

    let windowed_context = ContextBuilder::new()
        .build_windowed(wb, &el)
        .unwrap();

    let windowed_context = unsafe { windowed_context.make_current().unwrap() };

    println!("Pixel format of the window's GL context: {:?}", windowed_context.get_pixel_format());

    gl::load_with(|symbol| windowed_context.get_proc_address(symbol) as *const _);

    // windowed_context.window().set_cursor_grab(true).unwrap();
    // windowed_context.window().set_cursor_visible(false);

    let mut shader = shader::Shader::new("main");

    shader.compile();

    let verticies: [f32; 24] = [
        -1.0, 1.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 0.0,

        -1.0, 1.0, 0.0, 1.0,
        1.0, -1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0
    ];

    let mut vbo: u32 = 0;
    let mut vao: u32 = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (verticies.len() * std::mem::size_of::<f32>()) as isize, verticies.as_ptr() as *const _, gl::STATIC_DRAW);

        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 4 * std::mem::size_of::<f32>() as i32, std::ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, 4 * std::mem::size_of::<f32>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(1);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);
    }

    let (mut width, mut height): (u32, u32) = windowed_context.window().inner_size().into();

    width = width / 8;
    height = height / 8;

    let buffer_size = (width * height) as usize;

    let mut pbo: u32 = 0;
    let mut mapped_ptr: *mut c_void = ptr::null_mut();

    unsafe {
        gl::GenBuffers(1, &mut pbo);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, pbo);
        gl::BufferStorage(
            gl::PIXEL_UNPACK_BUFFER,
            (buffer_size * std::mem::size_of::<f32>()) as isize,
            ptr::null(),
            gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT
        );
        
        mapped_ptr = gl::MapBufferRange(
            gl::PIXEL_UNPACK_BUFFER,
            0,
            (buffer_size * std::mem::size_of::<f32>()) as isize,
            gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_FLUSH_EXPLICIT_BIT
        );
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
    }

    let mut texture: u32 = 0;

    unsafe {
        gl::GenTextures(1, &mut texture);
        gl::BindTexture(gl::TEXTURE_2D, texture);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);

        gl::TexImage2D(gl::TEXTURE_2D, 0, gl::R32F as i32, width as i32, height as i32, 0, gl::RED, gl::FLOAT, std::ptr::null());
        gl::BindTexture(gl::TEXTURE_2D, 0);
    }

    let mut data: Vec<f32> = vec![0.0; (width * height) as usize];

    let mut fluid_solver = FluidSolver::new(width, height);

    // for i in 0..data.len() {
    //     data[i] = rand::thread_rng().gen_range(0..2) as f32;
    // }

    let mut last_frame = Instant::now();

    let mut start_frame: Instant = Instant::now();

    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::LoopDestroyed => return,
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::MouseInput { state, button, .. } => {
                    match button {
                        glutin::event::MouseButton::Left => {
                            fluid_solver.mouse_down = state == ElementState::Pressed;
                        },
                        glutin::event::MouseButton::Right => {
                            fluid_solver.right_mouse_down = state == ElementState::Pressed;
                        },
                        _ => ()
                    }
                },
                WindowEvent::CursorMoved { position, .. } => {
                    // Convert screen coordinates to fluid grid coordinates
                    let x = position.x as f32 / 8.0;
                    let y = (height as f32) - (position.y as f32 / 8.0);
                    
                    // Store previous position before updating
                    fluid_solver.prev_mouse_pos = fluid_solver.mouse_pos;
                    fluid_solver.mouse_pos = (x, y);
                },
                _ => (),
            },
            Event::MainEventsCleared => {
                let current_frame = Instant::now();
                let delta_time = current_frame.duration_since(last_frame).as_secs_f32();
                last_frame = current_frame;

                windowed_context.window().request_redraw();
            },
            Event::RedrawRequested(_) => {
                // Replace the mouse interaction code in RedrawRequested with:
                let (mouse_x, mouse_y) = fluid_solver.mouse_pos;
                let (prev_x, prev_y) = fluid_solver.prev_mouse_pos;

                // Calculate velocity from mouse movement with speed scaling
                let dx = mouse_x - prev_x;
                let dy = mouse_y - prev_y;
                let speed = (dx * dx + dy * dy).sqrt();
                let vel_x = dx * speed * 5.0; // Increased velocity multiplier
                let vel_y = dy * speed * 5.0;

                if (fluid_solver.mouse_down || fluid_solver.right_mouse_down) && 
                mouse_x >= 0.0 && mouse_x < width as f32 && 
                mouse_y >= 0.0 && mouse_y < height as f32 {
                    
                    // Add velocity in a 3x3 area around mouse for smoothness
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let px = (mouse_x as i32 + dx) as u32;
                            let py = (mouse_y as i32 + dy) as u32;
                            
                            if px < width && py < height {
                                // Only add density with left click
                                if fluid_solver.mouse_down {
                                    fluid_solver.add_density(px, py, 25.0);
                                }
                                fluid_solver.add_velocity(px, py, vel_x, vel_y);
                            }
                        }
                    }
                }

                fluid_solver.step(0.01);
                fluid_solver.render(&mut data);

                unsafe {
                    gl::ClearColor(0.0, 0.0, 0.0, 1.0);
                    gl::Clear(gl::COLOR_BUFFER_BIT);

                    shader.use_program();

                    ptr::copy_nonoverlapping(data.as_ptr() as *const c_void, mapped_ptr, data.len() * std::mem::size_of::<f32>());

                    gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, pbo);
                    gl::FlushMappedBufferRange(
                        gl::PIXEL_UNPACK_BUFFER,
                        0,
                        (buffer_size * std::mem::size_of::<f32>()) as isize
                    );

                    gl::BindTexture(gl::TEXTURE_2D, texture);

                    gl::TexImage2D(gl::TEXTURE_2D, 0, gl::R32F as i32, width as i32, height as i32, 0, gl::RED, gl::FLOAT, std::ptr::null());

                    gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);

                    shader.set_int("grid", 0);
                    
                    gl::ActiveTexture(gl::TEXTURE0);
                    gl::BindTexture(gl::TEXTURE_2D, texture);
                    gl::BindVertexArray(vao);
                    gl::DrawArrays(gl::TRIANGLES, 0, 6);
                }

                windowed_context.swap_buffers().unwrap();

                println!("FPS: {}", 1.0 / last_frame.elapsed().as_secs_f32());
            },
            _ => (),
        }
    });
}
