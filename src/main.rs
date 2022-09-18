#![allow(clippy::toplevel_ref_arg)]
mod ad;
mod example1;
mod example2;
mod playground;

fn main() {
    playground::run();
    example1::run();
    example2::run();
}
