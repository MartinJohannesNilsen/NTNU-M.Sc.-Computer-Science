extern crate nalgebra_glm as glm;
use gl::types::*;
use std::{ mem, ptr, os::raw::c_void, str };
use std::thread;
use std::sync::{Mutex, Arc, RwLock};

mod shader;
mod util;

use glutin::event::{Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState::{Pressed, Released}, VirtualKeyCode::{self, *}};
use glutin::event_loop::ControlFlow;

const SCREEN_W: u32 = 800;
const SCREEN_H: u32 = 600;

// == // Helper functions to make interacting with OpenGL a little bit prettier. You *WILL* need these! // == //
// The names should be pretty self explanatory
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}

// Get a null pointer (equivalent to an offset of 0)
// ptr::null()



// == // Modify and complete the function below for the first task
// unsafe fn FUNCTION_NAME(ARGUMENT_NAME: &Vec<f32>, ARGUMENT_NAME: &Vec<u32>) -> u32 { } 

unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>) -> u32 { 
    //Create VAO
    let mut vao = 0;
    gl::GenVertexArrays(1, &mut vao);
    assert_ne!(vao, 0); //Asserts not equals 0 for safety measures
    gl::BindVertexArray(vao);

    //Create VBO
    let mut vbo = 0;
    gl::GenBuffers(1, &mut vbo);
    assert_ne!(vbo, 0);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
    
    //Transfer data to GPU
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW
    );
    
    //Describe vertex attributes for interpreting the bytes correctly
    gl::VertexAttribPointer(
        0, //index
        3,
        gl::FLOAT,
        gl::FALSE,
        0,
        ptr::null()
    );
    gl::EnableVertexAttribArray(0); //Use the index above
    
    //Create index buffer
    let mut index_buffer = 0;
    gl::GenBuffers(1, &mut index_buffer);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW
    );

    //return the integer ID of the created OpenGL VAO
    return vao
} 

unsafe fn create_vao_with_colors(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>) -> u32 { 
    //Create VAO
    let mut vao_id = 0;
    gl::GenVertexArrays(1, &mut vao_id);
    assert_ne!(vao_id, 0); //Asserts not equals 0 for safety measures
    gl::BindVertexArray(vao_id);

    //Create VBO
    let mut vbo_id = 0;
    gl::GenBuffers(1, &mut vbo_id);
    assert_ne!(vbo_id, 0);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo_id);
    
    //Transfer data to GPU
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW
    );
    
    //Describe vertex attributes for interpreting the bytes correctly
    gl::VertexAttribPointer(
        0, 3, gl::FLOAT, gl::FALSE, 0, ptr::null()
    );
    gl::EnableVertexAttribArray(0); //Use the index, the first parameter in VertexAttribPointer
    
    //Create index buffer
    let mut index_buffer_id = 0;
    gl::GenBuffers(1, &mut index_buffer_id);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer_id);
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW
    );

    //Create color buffer
    let mut color_buffer_id: u32 = 0;
    gl::GenBuffers(1, &mut color_buffer_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, color_buffer_id);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(colors),
        pointer_to_array(colors),
        gl::STATIC_DRAW
    );

    gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, 0, ptr::null());
    gl::EnableVertexAttribArray(1);

    //return the integer ID of the created OpenGL VAO
    return vao_id
} 

fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(false)
        .with_inner_size(glutin::dpi::LogicalSize::new(SCREEN_W, SCREEN_H));
    let cb = glutin::ContextBuilder::new()
        .with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    // Uncomment these if you want to use the mouse for controls, but want it to be confined to the screen and/or invisible.
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers. This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!("{}: {}", util::get_gl_string(gl::VENDOR), util::get_gl_string(gl::RENDERER));
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!("GLSL\t: {}", util::get_gl_string(gl::SHADING_LANGUAGE_VERSION));
        }

        // == // Set up your VAO here
        
        // Task 1
        let task1b_vertices: Vec<f32> = vec![
            //Top triangle
            -0.59, 0.61, 0.0,
            -0.0, 0.0, 0.0,
             0.6, 0.61, 0.0,    
        
            //Left triangle
            -0.61, -0.63, 0.0,
            -0.03, -0.03, 0.0,
            -0.61,  0.57, 0.0,    

            //Right triangle
            0.61, 0.57, 0.0,
            0.03, -0.03, 0.0,
            0.61, -0.63, 0.0,
        ];
        let task1b_indices: Vec<u32> = vec![
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        ];
        let task1b_colors: Vec<f32> = vec![
            1.0, 0.0, 1.0, 1.0, //Purple
            0.0, 1.0, 1.0, 1.0, //Blue
            1.0, 1.0, 0.0, 1.0, //Yellow

            1.0, 1.0, 0.0, 1.0, //Yellow
            0.0, 1.0, 1.0, 1.0, //Blue
            1.0, 0.0, 1.0, 1.0, //Purple

            1.0, 1.0, 0.0, 1.0, //Yellow
            0.0, 1.0, 1.0, 1.0, //Blue
            1.0, 0.0, 1.0, 1.0, //Purple
        ];

        // Task 2
        let task2_vertices: Vec<f32> = vec![
            //Left triangle
            -0.8,-0.8, 0.3,
             0.3, 0.0, 0.3,
            -0.8, 0.8, 0.3,     
            
            //Top triangle
            -0.8,  0.8, 0.0,
             0.0, -0.2, 0.0,
             0.8,  0.8, 0.0,   

            //Right triangle
             0.80, 0.8,-0.3,
            -0.30, 0.0,-0.3,
             0.80,-0.8,-0.3,
        ];
        let task2_indices: Vec<u32> = vec![
            0, 1, 2,
            3, 4, 5,    
            6, 7, 8,
        ];
        let task2_colors: Vec<f32> = vec![
            0.545, 0.000, 0.545, 0.5, //DarkMagenta
            0.545, 0.000, 0.545, 0.5, //DarkMagenta
            0.545, 0.000, 0.545, 0.5, //DarkMagenta

            0.196, 0.804, 0.196, 0.5, //LimeGreen
            0.196, 0.804, 0.196, 0.5, //LimeGreen
            0.196, 0.804, 0.196, 0.5, //LimeGreen

            0.941, 0.502, 0.502, 0.5, //LightCoral
            0.941, 0.502, 0.502, 0.5, //LightCoral
            0.941, 0.502, 0.502, 0.5, //LightCoral
        ];
        
        unsafe {
            create_vao_with_colors(&task1b_vertices, &task1b_indices, &task1b_colors);
        }

        // Basic usage of shader helper:
        // The example code below returns a shader object, which contains the field `.program_id`.
        // The snippet is not enough to do the assignment, and will need to be modified (outside of
        // just using the correct path), but it only needs to be called once
        //
        //     shader::ShaderBuilder::new()
        //        .attach_file("./path/to/shader.file")
        //        .link();
        let shader = unsafe {
            shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.vert")
                .attach_file("./shaders/simple.frag")
                .link()
        };

        unsafe {
            gl::UseProgram(shader.program_id);
        }

        // Used to demonstrate keyboard handling -- feel free to remove
        let mut _arbitrary_number = 0.0;

        let first_frame_time = std::time::Instant::now();
        let mut last_frame_time = first_frame_time;

        // Initialize vector containing position and orientation of the camera
        // Format: [trans(x), trans(y), trans(z), rot(y), rot(x)]
        let mut eta: Vec<f32> = vec![
            0.0, 0.0, -2.0, 0.0, 0.0
        ];

        // The main rendering loop
        loop {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = now;

            // Handle keyboard input
            // Translate up/down/right/left with WASD
            // Scale up/down with EF
            // Rotate with arrows
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::A => {
                            eta[0] += delta_time;
                        },
                        VirtualKeyCode::D => {
                            eta[0] -= delta_time;
                        },
                        VirtualKeyCode::S => {
                            eta[1] += delta_time;
                        },
                        VirtualKeyCode::W => {
                            eta[1] -= delta_time;
                        },
                        VirtualKeyCode::E => {
                            eta[2] -= delta_time;
                        },
                        VirtualKeyCode::F => {
                            eta[2] += delta_time;
                        },
                        VirtualKeyCode::Up => {
                            eta[3] -= delta_time;
                        },
                        VirtualKeyCode::Down => {
                            eta[3] += delta_time;
                        },VirtualKeyCode::Left => {
                            eta[4] -= delta_time;
                        },
                        VirtualKeyCode::Right => {
                            eta[4] += delta_time;
                        },


                        _ => { }
                    }
                }
            }
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {


                *delta = (0.0, 0.0);
            }

            unsafe {
                gl::ClearColor(1.0, 1.0, 1.0, 1.0); // White background
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // Issue the necessary commands to draw your scene here
                gl::DrawElements(gl::TRIANGLES, 16, gl::UNSIGNED_INT, ptr::null());

                // Task 4
                // Defines some transformation matrices
                let rotate_y: glm::Mat4 = glm::mat4(
                    eta[4].cos(), 0.0, eta[4].sin(), 0.0, 
                    0.0, 1.0, 0.0, 0.0, 
                    -eta[4].sin(), 0.0, eta[4].cos(), 0.0, 
                    0.0, 0.0, 0.0, 1.0,
                );
                let rotate_x: glm::Mat4 = glm::mat4(
                    1.0, 0.0, 0.0, 0.0, 
                    0.0, eta[3].cos(), -eta[3].sin(), 0.0, 
                    0.0, eta[3].sin(), eta[3].cos(), 0.0, 
                    0.0, 0.0, 0.0, 1.0,
                );
                let translate: glm::Mat4 = glm::mat4(
                    1.0, 0.0, 0.0, eta[0], 
                    0.0, 1.0, 0.0, eta[1], 
                    0.0, 0.0, 1.0, eta[2], 
                    0.0, 0.0, 0.0, 1.0,
                );

                // Perspective and create the uniform matrix
                let perspective_transform: glm::Mat4 = glm::perspective(1.0, 1.0, 1.0, 100.0);
                gl::UniformMatrix4fv(5, 1, 0, (perspective_transform * rotate_y * rotate_x * translate).as_ptr());

                // Issue the necessary commands to draw your scene here
                gl::DrawElements(gl::TRIANGLES, 9, gl::UNSIGNED_INT, ptr::null());
            }

            context.swap_buffers().unwrap();
        }
    });

    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events get handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                input: KeyboardInput { state: key_state, virtual_keycode: Some(keycode), .. }, .. }, .. } => {

                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        },
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle escape separately
                match keycode {
                    Escape => {
                        *control_flow = ControlFlow::Exit;
                    },
                    Q => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => { }
                }
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            },
            _ => { }
        } 
    });
}
