
use std::fs;
use std::path::Path;
use std::process::Command;
use futures::executor::block_on;
use futures::future::{self, Future};

fn main() {
    println!("cargo:rerun-if-changed=src/shaders");
    println!("cargo:rerun-if-changed=slang/build/Release/bin/slangc");
    println!("cargo:rerun-if-changed=build.rs");
    let slang_compiler = Path::new("./slang/build/Release/bin/slangc");
    let shaders_dir = Path::new("src/shaders");

    let mut tasks = Vec::new();

    if let Ok(entries) = fs::read_dir(shaders_dir) {
        let mut c = 0;
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("slang") {
                    c += 1;
                    let input_file = path.to_str().unwrap().to_string();
                    let output_file = path.with_extension("spv").to_str().unwrap().to_string();
                    let compiler_path = slang_compiler.to_path_buf();

                    let task = future::lazy(move |_| {
                        let status = Command::new(compiler_path)
                            .arg(&input_file)
                            .arg("-profile")
                            .arg("glsl_450")
                            .arg("-target")
                            .arg("spirv")
                            .arg("-o")
                            .arg(&output_file)
                            .arg("-entry")
                            .arg("main")
                            .status()
                            .expect("Failed to execute slangc");
                        if !status.success() {
                            panic!("Shader compilation failed for {}", input_file);
                        }

                        println!("cargo:rerun-if-changed={}", input_file);
                    });

                    tasks.push(task);
                }
            }
        }
        if c == 0 { panic!("No shaders found in src/shaders"); }
    }

    // Wait for all tasks to complete
    block_on(future::join_all(tasks));
}
