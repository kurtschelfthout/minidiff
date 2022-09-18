use std::rc::Rc;

use crate::ad::*;

fn f<T: VectorSpace>(t: &Dual<T>) -> Dual<T> {
    t.powi(2) + t + Dual::constant(1.0)
}

//wrapper
fn df(t: f64) -> f64 {
    let res = f(&Dual::new(t, 1.0));
    res.tangent
}

//mulitple outputs - works nicely
fn f_2out<T: VectorSpace>(t: &Dual<T>) -> (Dual<T>, Dual<T>) {
    let x = t.powi(2) + t + Dual::constant(1.0);
    let y = t.powi(2) + Dual::constant(3.0) * t + Dual::constant(2.0);
    (x, y)
}

fn df_2out(x: f64) -> (f64, f64) {
    let (dx, dy) = f_2out(&Dual::new(x, 1.0));
    (dx.tangent, dy.tangent)
}

//multiple inputs: bad - evaluates function multiple times!
fn f_3in<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>, z: &Dual<T>) -> Dual<T> {
    x * y.sin() + z * y
}

fn df_3in_v1(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r1 = f_3in(&Dual::new(x, 1.0), &Dual::constant(y), &Dual::constant(z));
    let r2 = f_3in(&Dual::constant(x), &Dual::new(y, 1.0), &Dual::constant(z));
    let r3 = f_3in(&Dual::constant(x), &Dual::constant(y), &Dual::new(z, 1.0));
    (r1.tangent, r2.tangent, r3.tangent)
}

// "full jacobian"
// fn f_3in3out<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>, z: &Dual<T>) -> (Dual<T>, Dual<T>, Dual<T>) {
//     (x * y, x * z, y * z)
// }

// fn df_3in3out(x: f64, y: f64, z: f64) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64)) {
//     let (r11, r12, r13) = f_3in3out(&Dual::new(x, 1.0), &Dual::constant(y), &Dual::constant(z));
//     let (r21, r22, r23) = f_3in3out(&Dual::constant(x), &Dual::new(y, 1.0), &Dual::constant(z));
//     let (r31, r32, r33) = f_3in3out(&Dual::constant(x), &Dual::constant(y), &Dual::new(z, 1.0));
//     (
//         (r11.tangent, r12.tangent, r13.tangent),
//         (r21.tangent, r22.tangent, r23.tangent),
//         (r31.tangent, r32.tangent, r33.tangent),
//     )
// }

// much better - f only called once - but memory is quadratic n_inputs x n_inputs
fn df_3in_v2(x: f64, y: f64, z: f64) -> [f64; 3] {
    let r = f_3in(
        &Dual::new(x, [1.0, 0.0, 0.0]),
        &Dual::new(y, [0.0, 1.0, 0.0]),
        &Dual::new(z, [0.0, 0.0, 1.0]),
    );
    r.tangent
}

fn df_3in_v3(x: f64, y: f64, z: f64) -> Vec<f64> {
    let r = f_3in(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
        &Dual::new(z, Rc::new(Delta::Var(2))),
    );
    let mut result = vec![0.0, 0.0, 0.0];
    eval_delta(1.0, &r.tangent, &mut result);
    result
}

fn f_sharing<'a>(x: &DualTrace<'a>, y: &DualTrace<'a>, z: &DualTrace<'a>) -> DualTrace<'a> {
    let ref y = x.sin() * y.sin() * z * z; // some big rhs
    y + y
}

fn f_sharing_bad<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>, z: &Dual<T>) -> Dual<T> {
    let ref y = x.sin() * y.sin() * z * z; // some big rhs
    y + y
}

fn df_sharing_bad(x: f64, y: f64, z: f64) -> Vec<f64> {
    let r = f_sharing_bad(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
        &Dual::new(z, Rc::new(Delta::Var(2))),
    );
    let mut result = vec![0.0, 0.0, 0.0];
    eval_delta(1.0, &r.tangent, &mut result);
    result
}

fn df_sharing(x: f64, y: f64, z: f64) -> Vec<f64> {
    let trace = Trace::new();
    let x = &trace.var(x);
    let y = &trace.var(y);
    let z = &trace.var(z);
    let dual_trace = f_sharing(x, y, z);
    eval(3, &dual_trace)
}

pub fn run() {
    let res = f(&Dual::new(10.0, 1.0));
    println!("{res:?}");

    let res = df(10.0);
    println!("{res:?}");

    let res = f_2out(&Dual::new(10.0, 1.0));
    println!("{res:?}");

    let res = df_2out(10.0);
    println!("{res:?}");

    let res = f_3in(
        &Dual::new(10.0, 1.0),
        &Dual::constant(5.0),
        &Dual::constant(1.0),
    );
    println!("{res:?}");

    let res = df_3in_v1(10.0, 5.0, 1.0);
    println!("{res:?}");

    let res = df_3in_v2(10.0, 5.0, 1.0);
    println!("{res:?}");

    let res = df_3in_v3(10.0, 5.0, 1.0);
    println!("{res:?}");

    let res = df_sharing_bad(1.0, 2.0, 3.0);
    println!("{res:?}");

    let res = df_sharing(1.0, 2.0, 3.0);
    println!("{res:?}");
}
