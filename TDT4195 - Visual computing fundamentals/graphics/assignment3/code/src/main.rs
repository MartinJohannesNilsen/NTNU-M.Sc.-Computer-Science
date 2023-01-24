extern crate nalgebra_glm as glm;
use gl::types::*;
use std::{ mem, ptr, os::raw::c_void, str };
use std::thread;
use std::sync::{Mutex, Arc, RwLock};

mod shader;
mod util;
mod mesh;
mod scene_graph;
mod toolbox;
use scene_graph::SceneNode;
use toolbox::Heading;
use toolbox::simple_heading_animation;

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

unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>, normals: &Vec<f32>) -> u32 { 
    //Create VAO
    let mut vao_id: u32 = 0;
    gl::GenVertexArrays(1, &mut vao_id);
    assert_ne!(vao_id, 0); //Asserts not equals 0 for safety measures
    gl::BindVertexArray(vao_id);

    //Create VBO
    let mut vbo_id: u32 = 0;
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
    let mut index_buffer_id: u32 = 0;
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

    //Create normals buffer
    let mut normal_buffer_id: u32 = 0;
    gl::GenBuffers(1, &mut normal_buffer_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, normal_buffer_id);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(normals),
        pointer_to_array(normals),
        gl::STATIC_DRAW
    );

    gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
    gl::EnableVertexAttribArray(2);

    //return the integer ID of the created OpenGL VAO
    return vao_id
} 

unsafe fn draw_scene(root: &SceneNode, view_projection_matrix: &glm::Mat4) {
    // Check if node is drawable, set uniforms, draw
    if (root.index_count > 0) {
        gl::UniformMatrix4fv(5, 1, 0, (view_projection_matrix * root.current_transformation_matrix).as_ptr()); //Model View Projection matrix
        gl::UniformMatrix4fv(6, 1, 0, (root.current_transformation_matrix).as_ptr());
        gl::BindVertexArray(root.vao_id);
        gl::DrawElements(gl::TRIANGLES, root.index_count, gl::UNSIGNED_INT, ptr::null());
    }

    // Recurse
    for &child in &root.children {
        draw_scene(&*child, view_projection_matrix);
    }
}

unsafe fn update_node_transformations(root: &mut SceneNode, transformation_so_far: &glm::Mat4) {
    // Construct the correct transformation matrix
    let origin_matrix = glm::mat4(
        1.0, 0.0, 0.0, root.reference_point[0],
        0.0, 1.0, 0.0, root.reference_point[1],
        0.0, 0.0, 1.0, root.reference_point[2],
        0.0, 0.0, 0.0, 1.0,
    );

    let rotate_x_matrix = glm::mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, root.rotation[0].cos(), -root.rotation[0].sin(), 0.0,
        0.0, root.rotation[0].sin(), root.rotation[0].cos(), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    let rotate_y_matrix = glm::mat4(
        root.rotation[1].cos(), 0.0, root.rotation[1].sin(), 0.0,
        0.0, 1.0, 0.0, 0.0,
        -root.rotation[1].sin(), 0.0, root.rotation[1].cos(), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    let rotate_z_matrix = glm::mat4(
        root.rotation[2].cos(), -root.rotation[2].sin(), 0.0, 0.0,
        root.rotation[2].sin(), root.rotation[2].cos(), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    let inverse_origin_matrix = glm::mat4(
        1.0, 0.0, 0.0, -root.reference_point[0],
        0.0, 1.0, 0.0, -root.reference_point[1],
        0.0, 0.0, 1.0, -root.reference_point[2],
        0.0, 0.0, 0.0, 1.0,
    );

    let translation_matrix = glm::mat4(
        1.0, 0.0, 0.0, root.position[0],
        0.0, 1.0, 0.0, root.position[1],
        0.0, 0.0, 1.0, root.position[2],
        0.0, 0.0, 0.0, 1.0,
    );
    // Update the node's transformation matrix
    root.current_transformation_matrix = transformation_so_far * translation_matrix * origin_matrix * rotate_x_matrix * rotate_y_matrix * rotate_z_matrix * inverse_origin_matrix;

    // Recurse
    for &child in &root.children {
        update_node_transformations(&mut *child,
        &root.current_transformation_matrix);
    }
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
        let terrain = mesh::Terrain::load("./resources/lunarsurface.obj");
        let terrain_vao = unsafe { create_vao(&terrain.vertices, &terrain.indices, &terrain.colors, &terrain.normals) };
        
        // Set up scene graph
        let mut root_node = SceneNode::new();
        let mut terrain_node = SceneNode::from_vao(terrain_vao, terrain.index_count);

        //Let's create some helicopters, shall we?
        let helicopter = mesh::Helicopter::load("./resources/helicopter.obj");
        let mut helicopter_body_nodes: Vec<scene_graph::Node> = Vec::new();
        let mut helicopter_tail_rotor_nodes: Vec<scene_graph::Node> = Vec::new();
        let mut helicopter_main_rotor_nodes: Vec<scene_graph::Node> = Vec::new();
        let n_helicopters = 5;

        for n in 0..n_helicopters {
            //Create VAO of parts of helicopter
            let helicopter_body_vao = unsafe { create_vao(&helicopter.body.vertices, &helicopter.body.indices, &helicopter.body.colors, &helicopter.body.normals) };
            let helicopter_main_vao = unsafe { create_vao(&helicopter.main_rotor.vertices, &helicopter.main_rotor.indices, &helicopter.main_rotor.colors, &helicopter.main_rotor.normals) };
            let helicopter_tail_vao = unsafe { create_vao(&helicopter.tail_rotor.vertices, &helicopter.tail_rotor.indices, &helicopter.tail_rotor.colors, &helicopter.tail_rotor.normals) };
            let helicopter_door_vao = unsafe { create_vao(&helicopter.door.vertices, &helicopter.door.indices, &helicopter.door.colors, &helicopter.door.normals) };

            //Create scene node
            let mut helicopter_body_node = SceneNode::from_vao(helicopter_body_vao, helicopter.body.index_count);
            let mut helicopter_main_rotor_node = SceneNode::from_vao(helicopter_main_vao, helicopter.main_rotor.index_count);
            let mut helicopter_tail_rotor_node = SceneNode::from_vao(helicopter_tail_vao, helicopter.tail_rotor.index_count);
            let mut helicopter_door_node = SceneNode::from_vao(helicopter_door_vao, helicopter.door.index_count);
            
            //Create scene graph
            helicopter_body_node.add_child(&helicopter_door_node);
            helicopter_body_node.add_child(&helicopter_tail_rotor_node);
            helicopter_body_node.add_child(&helicopter_main_rotor_node);
            terrain_node.add_child(&helicopter_body_node);
            
            //Set reference point
            helicopter_tail_rotor_node.reference_point = glm::vec3(0.35, 2.3, 10.4);
            
            //Add to vector
            helicopter_body_nodes.push(helicopter_body_node);
            helicopter_tail_rotor_nodes.push(helicopter_tail_rotor_node);
            helicopter_main_rotor_nodes.push(helicopter_main_rotor_node);
        };
        root_node.add_child(&terrain_node);
        
        // Add shaders
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
        // Format: [body_trans(x), body_trans(y), body_trans(z), body_rot(y), body_rot(x), door_slide]
        let mut eta: Vec<f32> = vec![
            0.0, 0.0, -2.0, 0.0, 0.0, 0.0
        ];

        // The main rendering loop
        loop {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = now;

            // Handle keyboard input with WASDEF and arrows, open/close doors with O/P
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::A => {
                            eta[0] += 20.0 * delta_time;
                        },
                        VirtualKeyCode::D => {
                            eta[0] -= 20.0 * delta_time;
                        },
                        VirtualKeyCode::F => {
                            eta[1] += 20.0 * delta_time;
                        },
                        VirtualKeyCode::E => {
                            eta[1] -= 20.0 * delta_time;
                        },
                        VirtualKeyCode::W => {
                            eta[2] += 20.0 * delta_time;
                        },
                        VirtualKeyCode::S => {
                            eta[2] -= 20.0 * delta_time;
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
                        VirtualKeyCode::O => {
                            eta[5] -= delta_time;
                        },
                        VirtualKeyCode::P => {
                            eta[5] += delta_time;
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
                gl::ClearColor(0.9, 0.9, 0.9, 1.0); // White sky
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

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
                let perspective_transform: glm::Mat4 = glm::perspective(1.0, 1.0, 1.0, 1000.0);
                let transform_matrix: glm::Mat4 = perspective_transform * rotate_y * rotate_x * translate;
                
                // Modify scene graph and animate
                for i in 0..helicopter_body_nodes.len() {
                    helicopter_main_rotor_nodes[i].rotation.y += delta_time * 20.0;
                    helicopter_tail_rotor_nodes[i].rotation.x += delta_time * 20.0;
                    let animation = simple_heading_animation(elapsed + 0.8 * i as f32);
                    helicopter_body_nodes[i].position.y = 5.0;
                    helicopter_body_nodes[i].position.x = animation.x * 2.0;
                    helicopter_body_nodes[i].position.z = animation.z * 2.0;
                    helicopter_body_nodes[i].rotation.z = animation.roll;
                    helicopter_body_nodes[i].rotation.y = animation.yaw;
                    helicopter_body_nodes[i].rotation.x = animation.pitch;
                    helicopter_body_nodes[i][0].position.z = eta[5];
                };

                //Single helicopter
                /*
                let animation = simple_heading_animation(elapsed);
                helicopter_main_rotor_node.rotation.y += delta_time * 20.0;
                helicopter_tail_rotor_node.rotation.x += delta_time * 20.0;
                helicopter_body_node.position.y = 5.0;
                helicopter_body_node.position.x = animation.x;
                helicopter_body_node.position.z = animation.z;
                helicopter_body_node.rotation.z = animation.roll;
                helicopter_body_node.rotation.y = animation.yaw;
                helicopter_body_node.rotation.x = animation.pitch;
                */
                
                // Issue the necessary commands to draw your scene here
                let identity: glm::Mat4 = glm::identity();
                update_node_transformations(&mut root_node, &identity);
                draw_scene(&root_node, &transform_matrix);
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
